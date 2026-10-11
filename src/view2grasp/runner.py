from dataclasses import asdict,replace
from pathlib import Path
import json,time,traceback,hashlib,copy
import zipfile
import numpy as np
from .sim import Simulator,INSTRUCTIONS
from .types import EpisodeRecord
from .geometry import cloud,fit_table,grasp_candidates,pixel_point,project,unproject,part_cloud
from .mapping import DepthMap,UNKNOWN
from .planning import RobotModel,Planner,time_parameterize
from .vlm import VLM
from .evaluation import Evaluator
from .recording import save_observation,save_episode
from .config import ExperimentConfig,fingerprint
from .view_advisor import nominal_grasp,swept_unknown,suggest_views
from .reachability_advisor import suggest_reachable_views
from .observation_policy import tabletop_view
from .observation_crop import crop_observation
from .grounding_refinement import refine_grasp_grounding
from .grounding import ContactAssociationError,GroundingResolver,track_surface
from .grasp_semantics import check_vertical_part_midpoint
from .annotation_diagnostic import build_annotation_diagnostic
from .placement import tabletop_placement
from .transfer_recovery import different_tabletop_view,plan_transfer_attempt,tabletop_observation_budget


def leading_shell(points,resolution=.005):
    """Highest observed sweep point per XY voxel for upward-motion visibility."""
    points=np.asarray(points)
    if not len(points):return points.reshape(0,3)
    keys=np.round(points[:,:2]/resolution).astype(np.int32)
    order=np.argsort(points[:,2])[::-1]
    _,first=np.unique(keys[order],axis=0,return_index=True)
    return points[order[first]]


def _current_frame_reselection_reason(ranking):
    """A bounded search limitation, never a global infeasibility certificate."""
    if not isinstance(ranking,dict) or ranking.get('status')!='complete':return None
    total=ranking.get('input_candidates',0)
    if total<=0 or ranking.get('checked_candidates')!=total or ranking.get('incomplete_candidates')!=0:
        return None
    clear=ranking.get('nominal_clear_candidates',0)
    blocked=ranking.get('nominal_blocked_candidates',0)
    if clear+blocked!=total:return None
    if blocked==total:return 'all nominal gripper approaches intersect observed obstacles'
    # A truncated priority subset or interrupted IK search cannot establish
    # that all observed-clear hypotheses failed, even within this candidate set.
    if (clear>0 and ranking.get('selected_priority_candidates')==clear and
            ranking.get('priority_candidates_completed')==clear and
            ranking.get('priority_pregrasp_ik_attempts')==8*clear and
            ranking.get('priority_pregrasp_ik_solutions')==0):
        return 'all nominally unblocked candidates exhausted pregrasp IK seeds'
    return None


def _validate_current_frame_reselection(obs,decision):
    if decision.frame_id!=obs.frame_id or decision.camera_id!=obs.camera_id:
        raise ValueError('current-frame reselection returned stale frame or camera')
    if decision.action not in ('grasp','observe','abort'):
        raise ValueError('current-frame reselection requires grasp, observe or abort')
    if decision.action=='grasp':
        roles={g.role for g in decision.groundings if g.source=='vlm' and g.kind!='candidate'
            and g.frame_id==obs.frame_id and g.camera_id==obs.camera_id}
        if not {'object','grasp_part'}<=roles:
            raise ValueError('current-frame reselection requires fresh explicit object and grasp_part proposals')


def run_episode(task="block",seed=0,strategy="active",root="runs",replay=None,config=None):
    config=config or ExperimentConfig()
    out=Path(root)/f"{task}_{strategy}_{seed:04d}"
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Refusing to overwrite complete or partial episode: {out}")
    out.mkdir(parents=True,exist_ok=True)
    (out/"manifest.json").write_text(json.dumps({"source_sha256":fingerprint(config),"task":task,"seed":seed,
        "strategy":strategy,"online":replay is None,**config.model_dump(),"approach_cone_degrees":15},indent=2))
    package=Path(__file__).parent;repository=package.parent.parent
    with zipfile.ZipFile(out/'source_snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package.glob('*.py')):archive.write(path,'src/view2grasp/'+path.name)
        for name in ('uv.lock','pyproject.toml','LICENSE'):
            if (repository/name).exists():archive.write(repository/name,name)
        archive.writestr('episode_config.json',config.model_dump_json(indent=2))
    record=EpisodeRecord(task,seed,strategy,online=replay is None)
    # Every pre-close branch shares this limit, including residual recovery and
    # retries. Keep at least the initial observation for small smoke-test budgets.
    phase_budget=tabletop_observation_budget(task,config.max_observations,config.post_lift_observations)
    reserve_after_close=phase_budget.postclose_observations
    pregrasp_limit=phase_budget.pregrasp_limit
    events=[]; history=[]; start=time.monotonic(); sim=None; vlm=None
    robot_depths={}
    rng=np.random.default_rng(seed)
    def event(kind,**kwargs):
        item={"elapsed":time.monotonic()-start,"kind":kind,**kwargs}; events.append(item)
        print(json.dumps(item,ensure_ascii=False),flush=True)
    try:
        sim=Simulator(task,seed,config.image_size)
        evaluator=Evaluator(sim); sim.evaluator=evaluator
        robot=RobotModel(sim.robot_xml,sim.joint_names,sim.eef_name)
        sim.validate_initial_proxies(robot.points(sim.q,sim.gripper_q)[0])
        map_=DepthMap(config.voxel_resolution_m); map_.certify_home(robot.points(sim.q,sim.gripper_q)[0],robot.volumes(sim.q,sim.gripper_q))
        planner=Planner(robot,map_,rng)
        planner.gripper=sim.gripper_q
        vlm=VLM(out/"vlm",replay=replay)
        resolver=GroundingResolver.from_config(config)
        event('perception_backend',sam_available=resolver.segmenter is not None,reason=resolver.sam_unavailable_reason)
        def observe(*,after_close=False):
            limit=config.max_observations if after_close else pregrasp_limit
            if record.observations>=limit:
                raise RuntimeError('observation budget exhausted for '+('episode' if after_close else 'pre-grasp mapping'))
            obs=sim.observe(); record.observations+=1
            save_observation(out/"observations",obs)
            self_depth=robot.depth(obs);robot_depths[obs.frame_id]=self_depth
            map_.integrate(obs,robot.points(obs.q,obs.gripper_q)[0],self_depth,
                robot.volumes(obs.q,obs.gripper_q),robot.visual_volumes(obs.q,obs.gripper_q))
            return obs
        def execute(path,grip,phase):
            sim.phase=phase
            np.save(out/f"path_{len(events):03d}_{phase}.npy",path)
            event("execute",phase=phase,waypoints=len(path))
            for q in time_parameterize(path): sim.step(q,grip)
            for _ in range(8): sim.step(path[-1],grip)
            if np.max(np.abs(sim.q-path[-1]))>.12: raise RuntimeError("execution: joint tracking error")
        obs=observe(); initial_view=obs.view
        map_.set_table(fit_table(cloud(obs),rng))
        if strategy=='active':
            timer=time.monotonic();nominal=None if task=='mug' else nominal_grasp(robot,map_,obs)
            if nominal is not None:
                unknown=swept_unknown(robot,map_,obs,nominal)
                # The handle grasp's large wrist turn needs two complementary
                # pre-grasp coverage views to make the strict RRT corridor
                # observable. The final frame can still refresh the moved
                # gripper / payload. Remaining unobserved projection gaps are
                # rejected, never completed by neighbor voting.
                initial_budget=1
                advice=suggest_views(robot,obs,unknown,budget=max(0,min(initial_budget,config.max_observations-4)))
                history.append({'coverage_advice':advice})
                (out/'view_advice.json').write_text(json.dumps(advice,indent=2))
                event('view_advice',**advice)
            elif task=='mug':
                # Robot-sweep coverage advice can be orthogonal to a viable
                # handle ray. Let the VLM choose one broadside from the actual
                # overview; targeted map recovery follows after grounding.
                history.append({'feedback':
                    'No coverage-only view is prescribed for handle grounding. Infer the handle plane from this image and choose the reachable -X-side broadside described in the task.'})
            record.planning_seconds+=time.monotonic()-timer
        pending_decision=None;grounding_repairs={};contact_association_frames=set()
        current_frame_reselection_used=False
        approach_reselection_used=False
        grasp_refinement_frames=set()
        def contact_annotation(observation,grounded_decision,original_object,original_part,purpose,*,source_supported=False):
            # One annotation transaction per frame. A second correction is
            # allowed only after measurable geometric progress; no new image.
            if task!='occluded' or observation.frame_id in contact_association_frames:
                return None
            contact_association_frames.add(observation.frame_id)
            identities={g.target_id for g in grounded_decision.groundings
                        if g.role in ('object','grasp_part','grasp_contact')}
            identity=next(iter(identities)) if identities else 'target'
            crop=None;annotation_observation=observation
            image_context={'kind':'acquired_observation','source_frame_id':observation.frame_id,
                'source_camera_id':observation.camera_id,'source_shape_hw':list(observation.depth.shape),
                'coordinates':'normalized_to_source_pixel_centers','new_observation':False}
            try:
                crop=crop_observation(observation,original_object,
                    robot_depth=robot_depths.get(observation.frame_id))
                annotation_observation=crop.observation;image_context=crop.provenance()
            except ValueError as exc:
                event('contact_annotation_crop_unavailable',frame_id=observation.frame_id,reason=str(exc))
            history.append({'contact_reassociation':{
                'purpose':purpose+'; annotate the SAME object and originally permitted grasp part, preserving the stored grasp ray and candidates.',
                'frame_id':observation.frame_id,'camera_id':observation.camera_id,'target_id':identity,
                'original_grounding_frame':grounded_decision.frame_id,
                'image_context':image_context,
                'original_part':grounded_decision.part or (
                    'the originally requested grasp part' if any(g.role=='grasp_part' for g in grounded_decision.groundings)
                    else 'the whole target object'),
                'instructions':'Return observe with explicit tight object AND grasp_part polygon regions for THIS supplied image, using the same target_id. All normalized coordinates refer to the supplied image, including when image_context identifies a crop with source bounds. The projected reference only selects image context and does not prove that the object or part is visible. Exclude neighboring objects, support, robot, and holes. Do not substitute a box or point. This is annotation only: keep the current azimuth/elevation/distance; no camera command from this reply will execute. If the part cannot be identified, omit the regions and contact permission will remain unchanged.'}})
            annotation_request=copy.deepcopy(history[-1]['contact_reassociation'])
            previous_diagnostics=None;annotation_diagnostic=None
            if crop is not None:
                # Small distant targets may occupy only a few raw pixels.
                # Supply a coordinate-labelled magnification from the first
                # annotation, without adding scene evidence or semantic suggestions.
                annotation_diagnostic=build_annotation_diagnostic(annotation_observation,{
                    'frame_id':observation.frame_id,'camera_id':observation.camera_id,
                    'coordinate_space':'supplied_crop_normalized_pixel_centers',
                    'image_shape_hw':list(annotation_observation.depth.shape),
                    'rejected_samples':[]})
                event('contact_annotation_diagnostic',frame_id=observation.frame_id,
                    repair=0,provenance=annotation_diagnostic.provenance())
            for repair in range(3):
                try:
                    annotation=vlm.decide(annotation_observation,INSTRUCTIONS[task],history,force_observe=True,
                        observe_purpose='annotation',derived_image_context=image_context if crop is not None else None,
                        annotation_diagnostic=annotation_diagnostic)
                    raw_annotation=annotation.model_dump()
                    event('contact_annotation_decision',image_context=image_context,repair=repair,**raw_annotation)
                    if crop is not None:annotation=crop.restore_decision(annotation)
                    event('contact_association_decision',preserved_grounding_frame=grounded_decision.frame_id,
                        image_context=image_context,restored_to_source=crop is not None,repair=repair,
                        **annotation.model_dump())
                    association=resolver.reassociate_contact(observation,annotation,original_object,original_part,
                        identity,map_.table_z,robot_depths.get(observation.frame_id),source_supported=source_supported)
                    break
                except (ValueError,RuntimeError) as exc:
                    diagnostics=getattr(exc,'diagnostics',None)
                    event('contact_association_rejected',frame_id=observation.frame_id,reason=str(exc),
                        repair=repair,diagnostics=diagnostics)
                    progressed=(previous_diagnostics is not None and diagnostics is not None and (
                        (previous_diagnostics['role']=='object' and diagnostics['role']=='grasp_part') or
                        (diagnostics['anchor_pixels'],-diagnostics['rejected_pixel_count'])>
                        (previous_diagnostics['anchor_pixels'],-previous_diagnostics['rejected_pixel_count'])))
                    if isinstance(exc,ContactAssociationError) and (repair==0 or (repair==1 and progressed)):
                        feedback=copy.deepcopy(diagnostics)
                        previous_diagnostics=diagnostics
                        if crop is not None:
                            h,w=observation.depth.shape;ch,cw=annotation_observation.depth.shape
                            offset=np.asarray(crop.bounds_xyxy[:2])
                            def to_crop(points):
                                return ((np.asarray(points)*[w-1,h-1]-offset)/[cw-1,ch-1]).tolist()
                            feedback['coordinate_space']='supplied_crop_normalized_pixel_centers'
                            feedback['image_shape_hw']=[ch,cw]
                            for sample in feedback['rejected_samples']:
                                sample['source_pixel']=sample.pop('pixel')
                                sample['normalized']=to_crop(sample['normalized'])
                            projection=feedback['reference_projection']
                            if projection['bbox_normalized'] is not None:
                                bounds=to_crop(np.asarray(projection['bbox_normalized']).reshape(2,2))
                                projection['bbox_normalized']=np.asarray(bounds).ravel().tolist()
                        # The auxiliary panel marks diagnostic pixel locations
                        # on the same acquired RGB. It never changes the image
                        # used by grounding or selects a replacement region.
                        annotation_diagnostic=None
                        try:
                            annotation_diagnostic=build_annotation_diagnostic(annotation_observation,feedback)
                            event('contact_annotation_diagnostic',frame_id=observation.frame_id,
                                repair=repair+1,provenance=annotation_diagnostic.provenance())
                        except ValueError as diagnostic_error:
                            event('contact_annotation_diagnostic_unavailable',frame_id=observation.frame_id,
                                reason=str(diagnostic_error))
                        history.append({'contact_reassociation':{**annotation_request,
                            'previous_annotation':raw_annotation,'geometry_rejection':feedback,
                            'instructions':annotation_request['instructions']+
                                ' Correct this SAME supplied image. A second correction is allowed only when geometric agreement improves; there are at most three annotation replies per physical frame. Reinspect the visible material using the rejected pixels and projection hypothesis; these are geometry diagnostics, not semantic labels. You may annotate a smaller clearly identifiable visible surface, but do not blindly delete rejected pixels or guess hidden material. If uncertain, omit regions. All thresholds and contact permissions are unchanged.'}})
                        event('contact_annotation_repair',frame_id=observation.frame_id,
                            image_context=image_context,feedback=feedback)
                        continue
                    history.append({'feedback':'Contact reassociation rejected; previous contact permission and original grasp ray remain unchanged: '+str(exc)})
                    return None
            report={**association.report,'purpose':purpose,'preserved_grounding_frame':grounded_decision.frame_id,
                'image_context':image_context}
            np.savez_compressed(out/f'contact_association_{observation.frame_id:03d}.npz',
                object_mask=association.object_mask,part_mask=association.part_mask,
                object_points=association.object_points,part_points=association.part_points)
            (out/f'contact_association_{observation.frame_id:03d}.json').write_text(json.dumps(report,indent=2))
            event('contact_association',**report)
            history.append({'contact_association_result':report,
                'feedback':'Current-frame annotation verified. Resume the surrounding mapping or execution transaction; the original grasp ray and remaining coverage advice are unchanged.'})
            return association
        def reassociate_contacts(observation,grounded_decision,original_object,original_part,target,target_part):
            association=contact_annotation(observation,grounded_decision,original_object,original_part,
                'pre-close contact recovery for the static scene',source_supported=True)
            if association is None:return target,target_part
            # Keep measured points exactly: rounding representatives could
            # discard the newly observed side endpoint needed by the map.
            target=np.unique(np.vstack([target,association.object_points]),axis=0)
            target_part=np.unique(np.vstack([target_part,association.part_points]),axis=0)
            map_.set_target(target_part)
            return target,target_part
        while record.observations<=pregrasp_limit and record.attempts<config.max_attempts:
            # Bounded development policy: a larger mug budget permits one
            # alternate semantic view if the first broadside is occluded. Keep
            # at least one pre-close mapping slot and all post-close slots.
            # Once grounded, mapping recovery still preserves the original ray.
            force_at=(min(3,max(2,pregrasp_limit-1),pregrasp_limit) if task=='mug'
                else min(config.max_observations-3,pregrasp_limit))
            if task=='mug':
                history.append({'initial_grounding_deadline':force_at,
                    'observations_used':record.observations,
                    'remaining_pregrasp_observations':pregrasp_limit-record.observations,
                    'reserved_post_close_observations':reserve_after_close,
                    'feedback':'Ground now if the solid handle is visible. Before the initial grounding deadline, an occluded view may be replaced by another actual view. After grounding, preserve its approach ray during mapping recovery.'})
            decision=pending_decision or vlm.decide(obs,INSTRUCTIONS[task],history,
                force_grasp=record.observations>=force_at or strategy!="active")
            pending_decision=None
            refinement_resolved=None
            save_observation(out/"observations",obs,decision)
            event("decision",**decision.model_dump())
            if (task=='mug' and config.refine_grasp_grounding and decision.action=='grasp'
                    and obs.frame_id not in grasp_refinement_frames):
                # This explicit semantic request precedes contact/ray creation.
                # It reuses acquired pixels, never annotation's observe action.
                grasp_refinement_frames.add(obs.frame_id)
                coarse=decision
                try:
                    decision,refinement_resolved=refine_grasp_grounding(obs,coarse,resolver,vlm,
                        INSTRUCTIONS[task],history,map_.table_z,robot_depth=robot_depths.get(obs.frame_id),
                        event=lambda kind,**data:event(kind,observations_used=record.observations,**data))
                    save_observation(out/'observations',obs,decision)
                except (ValueError,RuntimeError) as exc:
                    record.failure='grounding refinement: '+str(exc)
                    event('grounding_refinement_rejected',frame_id=obs.frame_id,
                        feedback=str(exc),coarse_decision=coarse.model_dump())
                    break
            if decision.action=="abort": record.failure="agent_abort: "+decision.reason; break
            if decision.action=="observe":
                if record.observations>=pregrasp_limit:
                    record.failure='observation budget reserved for post-grasp mapping'
                    event('observation_budget_reserved',observations=record.observations,
                        reserved_after_close=reserve_after_close)
                    break
                sim.move_camera(decision.azimuth,decision.elevation,decision.distance)
                # Preserve the agent's visual inference (for example the mug's
                # handle direction) as compact textual memory. Images are still
                # acquired only after moving the camera; no unseen frame enters
                # history.
                history.append({"view":obs.view,"decision":decision.model_dump(),
                    "feedback":"requested new observation"}); obs=observe(); continue
            timer=time.monotonic();candidate_search_completed=False;checks_before=planner.checks
            try:
                resolved=(refinement_resolved if refinement_resolved is not None else
                    resolver.resolve_grasp(obs,decision,map_.table_z,robot_depths.get(obs.frame_id)))
                decision=resolved.decision
                record.grounding_types.extend(resolved.report['kinds'])
                record.segmentation_calls=resolver.calls;record.perception_seconds=resolver.seconds
                from .recording import save_grounding_result
                save_grounding_result(out,obs,resolved)
                if task=='mug':
                    semantic_check=check_vertical_part_midpoint(obs,resolved.object_mask,
                        resolved.part_mask,decision.point)
                    event('grasp_semantic_check',**semantic_check)
                candidates,target=grasp_candidates(obs,decision,map_.table_z,robot.pose(obs.q)[1],robot.grasp_center_offset(),map_.surface,resolved.object_mask,resolved.part_mask)
                if not candidates:raise ValueError('no candidates from observed grasp geometry')
                target_part=part_cloud(obs,decision,target,resolved.part_mask)
                # The 2-D part box identifies the intended component, but a
                # broadside view may expose only its front strip. Associate the
                # locally connected, actually observed 3-D surfaces around the
                # click so both sides of a handle bar can be legitimate finger
                # contacts. This remains image/depth based and never uses IDs.
                anchor=candidates[0].surface_point
                explicit_part=any(g.role=='grasp_part' for g in decision.groundings)
                local_part=(np.empty((0,3)) if explicit_part else
                    target[np.linalg.norm(target-anchor,axis=1)<.04])
                if len(local_part):
                    target_part=np.vstack([target_part,local_part])
                    _,part_unique=np.unique(np.round(target_part/map_.resolution).astype(np.int32),axis=0,return_index=True)
                    target_part=target_part[part_unique]
                map_.set_target(target_part)
                original_object=target.copy();original_part=target_part.copy()
                np.savez_compressed(out/f"candidates_{obs.frame_id:03d}.npz",positions=np.array([g.position for g in candidates]),
                    rotations=np.array([g.rotation for g in candidates]),widths=np.array([g.width for g in candidates]),
                    ray_angles=np.array([g.ray_angle for g in candidates]),target=target,
                    surface_points=np.array([g.surface_point for g in candidates]),grasp_centers=np.array([g.grasp_center for g in candidates]),
                    grasp_center_offset=robot.grasp_center_offset())
                planning_rng_state=copy.deepcopy(planner.rng.bit_generator.state)
                chosen=planner.choose(obs.q,candidates,max_candidates=config.max_candidates,
                    time_budget=config.planning_seconds,prioritize_gripper_geometry=task=='mug')
                candidate_search_completed=True
                (out/f'planning_diagnostic_{obs.frame_id:03d}.json').write_text(json.dumps({
                    'failure_counts':planner.failure_counts,'closest_invalid':planner.closest_invalid,
                    'closest_by_failure':planner.closest_by_failure,
                    'geometry_ranking':getattr(planner,'geometry_ranking',None),
                    'recovery_unknown_points':len(planner.recovery_unknown()),'checks':planner.checks},indent=2))
                feedback=planner.last_failure if chosen is None else "planned"
            except ValueError as exc:
                chosen=None; feedback="grounding: "+str(exc)
                if hasattr(exc,'diagnostics'):
                    event('grounding_rejected',diagnostics=exc.diagnostics,reason=str(exc))
                    history.append({'grounding_rejection':exc.diagnostics})
            planning_elapsed=time.monotonic()-timer
            record.planning_seconds+=planning_elapsed
            checks_delta=planner.checks-checks_before
            event("planning",feedback=feedback,checks=planner.checks,checks_delta=checks_delta)
            history.append({"view":obs.view,"feedback":feedback})
            if chosen is None:
                record.failure="planning: "+feedback
                if record.observations>=pregrasp_limit:
                    if (strategy=='active' and feedback.startswith('grounding:')
                            and grounding_repairs.get(obs.frame_id,0)<2):
                        # Fixing coordinates on the existing image consumes no
                        # post-close observations. Never accept another camera
                        # request once the pre-grasp frame limit is reached.
                        grounding_repairs[obs.frame_id]=grounding_repairs.get(obs.frame_id,0)+1
                        corrected=vlm.decide(obs,INSTRUCTIONS[task],history,force_grasp=True)
                        event('final_frame_grounding_repair',repair=grounding_repairs[obs.frame_id],
                            **corrected.model_dump())
                        if corrected.action=='grasp':
                            pending_decision=corrected
                            continue
                    event('observation_budget_reserved',observations=record.observations,
                        reserved_after_close=reserve_after_close)
                    break
                pure_ik_failure=(candidate_search_completed and checks_delta==0 and
                        feedback=='pregrasp IK' and set(planner.failure_counts)=={'pregrasp IK'} and
                        planner.failure_counts['pregrasp IK']>0)
                geometry_reason=(_current_frame_reselection_reason(getattr(planner,'geometry_ranking',None))
                    if task=='mug' and candidate_search_completed else None)
                if (strategy=='active' and not approach_reselection_used and
                        (pure_ik_failure or geometry_reason is not None)):
                    # Check the complete nominal-geometry evidence before
                    # spending mapping views on arm UNKNOWN. This is a bounded
                    # search limitation, not proof that every ray is infeasible.
                    approach_reselection_used=True
                    reason=geometry_reason or 'all tested candidates failed pregrasp IK before spatial validity checks'
                    contact=np.asarray(anchor);ray=contact-obs.T_wc[:3,3]
                    ray=ray/np.linalg.norm(ray)
                    timer=time.monotonic()
                    kinematic_advice=suggest_reachable_views(robot,obs,contact,target_part,map_.surface,
                        budget=3,max_ik_calls=72,time_budget=min(4.,max(0.,config.planning_seconds-planning_elapsed)))
                    record.planning_seconds+=time.monotonic()-timer
                    advice={'purpose':'approach_reselection','frame_id':obs.frame_id,
                        'camera_id':obs.camera_id,'current_view':list(obs.view),
                        'reason':reason,'geometry_ranking':copy.deepcopy(getattr(planner,'geometry_ranking',None)),
                        'observed_contact_world_m':contact.tolist(),
                        'camera_to_contact_ray':ray.tolist(),'approach_cone_degrees':15,
                        'failure_counts':dict(planner.failure_counts),'checks_delta':checks_delta,
                        'generated_candidates':len(candidates),
                        'kinematic_view_advice':kinematic_advice,
                        'approach_reselection_count':1,
                        'current_frame_reselection_count':int(current_frame_reselection_used),
                        'total_reselections':1+int(current_frame_reselection_used),
                        'remaining_pregrasp_observations':pregrasp_limit-record.observations,
                        'reserved_post_close_observations':reserve_after_close,
                        'scope':'Observed contact and bounded robot-kinematics failures only; no map clearance or global infeasibility claim.',
                        'next_step':'Choose a different calibrated view for a NEW semantic grounding and approach ray. Seeing the handle does not establish robot reachability. Consider nearby alternatives to a self-occluded view, not only its opposite side. The initial grounding deadline does not prohibit this recovery. This is not mapping-only: after the new image, re-ground object and grasp_part from that image; never execute the old contact with a silently changed ray.'}
                    history.append({'approach_reselection':advice})
                    event('approach_reselection_advice',**advice)
                    next_decision=vlm.decide(obs,INSTRUCTIONS[task],history,force_observe=True,
                        observe_purpose='approach_reselection')
                    event('approach_reselection_decision',**next_decision.model_dump())
                    if next_decision.action!='observe':break
                    selected=np.array([next_decision.azimuth,next_decision.elevation,next_decision.distance])
                    delta=selected-np.asarray(obs.view);delta[0]=(delta[0]+180)%360-180
                    if np.all(np.abs(delta)<1e-6):
                        record.failure='planning: '+feedback+'; approach recovery repeated the current view'
                        break
                    sim.move_camera(*selected)
                    history.append({'view':obs.view,'decision':next_decision.model_dump(),
                        'feedback':'Acquiring a new semantic view after bounded approach failure; retire the old approach ray and re-ground from the next current image.'})
                    obs=observe()
                    continue
                recovery=None
                recovery_points=planner.recovery_unknown() if candidate_search_completed else np.empty((0,3))
                if candidate_search_completed and not len(recovery_points): recovery_points=planner.best_unknown
                if candidate_search_completed and (recovery_points is None or not len(recovery_points)) and planner.closest_invalid is not None and planner.closest_invalid.get('unknown'):
                    recovery_points=np.asarray(planner.closest_invalid['points'])
                # Fallback diagnostic samples may mix UNKNOWN and OCCUPIED.
                # Additional coverage can address only genuinely unknown space;
                # an observed obstacle needs a different grasp/path instead.
                if recovery_points is not None and len(recovery_points):
                    recovery_points=np.asarray(recovery_points)
                    recovery_points=recovery_points[map_.classify(recovery_points)==UNKNOWN]
                if (candidate_search_completed and approach_reselection_used and
                        (pure_ik_failure or geometry_reason is not None) and
                        (recovery_points is None or not len(recovery_points))):
                    # Do not let the old generic observation fallback request
                    # another semantic view for the same exhausted recovery.
                    # Same-frame grounding repairs never reach this gate.
                    record.failure='planning: '+feedback+'; semantic view reselection limit reached'
                    event('approach_reselection_limit',reason=geometry_reason or 'pure pregrasp IK',
                        approach_reselection_count=1,
                        current_frame_reselection_count=int(current_frame_reselection_used),
                        total_reselections=1+int(current_frame_reselection_used))
                    break
                if recovery_points is not None and len(recovery_points):
                    # Closing moves both the gripper and the target. A single
                    # orbit view cannot in general expose the whole arm corridor
                    # and the leading payload shell (the block run left two arm
                    # and 34 payload voxels unseen). Spend only the shared
                    # pre-close allowance; residual retries obey the same cap.
                    # Advice covers the currently rejected points, not every
                    # later approach/RRT sample. Keep one pre-close frame for
                    # residuals discovered by replanning on the actual RGB-D.
                    remaining_pregrasp=pregrasp_limit-record.observations
                    recovery_budget=max(1,remaining_pregrasp-1)
                    recovery=suggest_views(robot,obs,recovery_points,budget=recovery_budget,
                        focus_points=target_part,observed_points=map_.surface)
                    history.append({'coverage_advice':recovery,'feedback':
                        'Targeted recovery advice for the specific robot points that remained unknown.'})
                    event('recovery_view_advice',**recovery)
                elif 'target interference' in feedback:
                    history.append({'feedback':
                        'The non-contact finger structure would hit the target. Reobserve the mug handle and ground the midpoint of the outermost vertical bar, away from the rim and both junctions.'})
                if strategy!="active":
                    sim.move_camera(*tabletop_view(strategy,initial_view,rng))
                # Active VLM chooses the next move from the same image and failure feedback.
                if strategy=="active":
                    steps=1 if recovery is None else min(len(recovery.get('views',[])),
                        recovery_budget)
                    recovered=False
                    for recovery_index in range(steps):
                        # Unknown-space recovery is mapping-only. A grasp from a
                        # different recovery frame would silently replace the
                        # frozen approach ray and make the representation rule
                        # test-dependent. Geometric/contact failures without
                        # recovery points may still request a corrected ground.
                        next_decision=vlm.decide(obs,INSTRUCTIONS[task],history,
                            force_observe=recovery is not None)
                        event("recovery_decision",recovery_index=recovery_index+1,**next_decision.model_dump())
                        if recovery is None and next_decision.action=='grasp' and grounding_repairs.get(obs.frame_id,0)<2:
                            grounding_repairs[obs.frame_id]=grounding_repairs.get(obs.frame_id,0)+1
                            pending_decision=next_decision
                            break
                        if next_decision.action!="observe": break
                        sim.move_camera(next_decision.azimuth,next_decision.elevation,next_decision.distance)
                        history.append({'view':obs.view,'decision':next_decision.model_dump(),
                            'feedback':'completed one mapping-only recovery observation; choose an unvisited remaining coverage view'})
                        obs=observe();recovered=True
                        newly_seen=(resolver.track(obs,target,robot_depths.get(obs.frame_id))
                            if task!='occluded' and 'target' in locals() else None)
                        if newly_seen is not None:
                            combined=np.vstack([target,newly_seen])
                            _,unique=np.unique(np.round(combined/map_.resolution).astype(np.int32),axis=0,return_index=True)
                            target=combined[unique]
                        if recovery is not None:
                            # Every acquired mapping image can expose a
                            # different target side. Verify it while available;
                            # the final view alone may hide those old endpoints.
                            # References stay fixed and the grasp ray is kept.
                            target,target_part=reassociate_contacts(obs,decision,
                                original_object,original_part,target,target_part)
                    if pending_decision is not None:continue
                    if not recovered: break
                    if recovery is None:
                        # A geometric/contact failure needs a new current-frame
                        # grounding and therefore returns through the VLM loop.
                        continue
                    # The scene has not moved. Preserve the VLM-selected point,
                    # camera ray and candidates; the extra view exists to reveal
                    # planning space, not silently change the grasp direction.
                    retry_timer=time.monotonic()
                    # Reuse the exact redundant-IK and RRT samples whose rejected
                    # workspace points produced the recovery advice. Otherwise a
                    # fresh random elbow branch creates new unseen space.
                    planner.rng.bit_generator.state=copy.deepcopy(planning_rng_state)
                    chosen=planner.choose(obs.q,candidates,max_candidates=config.max_candidates,
                        time_budget=config.planning_seconds,prioritize_gripper_geometry=task=='mug')
                    retry_elapsed=time.monotonic()-retry_timer
                    record.planning_seconds+=retry_elapsed
                    retry_feedback=planner.last_failure if chosen is None else "planned"
                    (out/f'planning_diagnostic_{obs.frame_id:03d}_recovery.json').write_text(json.dumps({
                        'failure_counts':planner.failure_counts,'closest_invalid':planner.closest_invalid,
                        'closest_by_failure':planner.closest_by_failure,
                        'geometry_ranking':getattr(planner,'geometry_ranking',None),
                        'recovery_unknown_points':len(planner.recovery_unknown()),'checks':planner.checks},indent=2))
                    event("recovery_planning",feedback=retry_feedback,checks=planner.checks,
                        preserved_grounding_frame=decision.frame_id)
                    if chosen is None:
                        record.failure="planning: "+retry_feedback
                        ranking=getattr(planner,'geometry_ranking',None)
                        reselection_reason=_current_frame_reselection_reason(ranking)
                        if (task=='mug' and strategy=='active' and not current_frame_reselection_used and
                                obs.frame_id!=decision.frame_id and record.observations<pregrasp_limit and
                                reselection_reason is not None):
                            # This image has already been acquired and integrated.
                            # Explicitly end the old ray transaction before asking
                            # for fresh semantics; retain the final mapping slot.
                            current_frame_reselection_used=True
                            timer=time.monotonic()
                            kinematic_advice=suggest_reachable_views(robot,obs,candidates[0].surface_point,
                                target_part,map_.surface,budget=3,max_ik_calls=72,
                                time_budget=min(4.,max(0.,config.planning_seconds-retry_elapsed)))
                            record.planning_seconds+=time.monotonic()-timer
                            advice={'purpose':'current_frame_reselection','frame_id':obs.frame_id,
                                'camera_id':obs.camera_id,'current_view':list(obs.view),
                                'retired_grounding_frame_id':decision.frame_id,
                                'reason':reselection_reason,'geometry_ranking':copy.deepcopy(ranking),
                                'failure_counts':dict(planner.failure_counts),
                                'previous_observed_contact_world_m':candidates[0].surface_point.tolist(),
                                'retired_camera_to_contact_ray':candidates[0].approach.tolist(),
                                'kinematic_view_advice':kinematic_advice,
                                'approach_reselection_count':int(approach_reselection_used),
                                'current_frame_reselection_count':1,
                                'total_reselections':1+int(approach_reselection_used),
                                'remaining_pregrasp_observations':pregrasp_limit-record.observations,
                                'reserved_post_close_observations':reserve_after_close,
                                'acquires_new_observation':False,
                                'scope':'Observed geometry and bounded candidate-search evidence only; '
                                    'nominal collision and failed IK seeds do not prove global infeasibility.',
                                'next_step':'End the previous mapping-only grasp transaction. Inspect this CURRENT '
                                    'already-acquired image and freshly ground the complete visible object and '
                                    'requested grasp_part to form a NEW camera-to-contact ray. Do not copy old '
                                    'pixels or execute the prior contact with a changed ray. If semantics are '
                                    'uncertain, use the bounded robot-kinematics view advice to select an actual '
                                    'observation within the remaining allowance, or abort. Semantic reselection '
                                    'may revisit a useful earlier view; mapping-only novelty is not a requirement. '
                                    'View advice does not establish visibility or map clearance. All ordinary '
                                    'grounding, candidate, UNKNOWN, collision and path checks remain required.'}
                            history.append({'current_frame_reselection':advice})
                            event('current_frame_reselection_advice',**advice)
                            fresh=vlm.decide(obs,INSTRUCTIONS[task],history,current_frame_reselection=advice)
                            event('current_frame_reselection_decision',**fresh.model_dump())
                            try:_validate_current_frame_reselection(obs,fresh)
                            except ValueError as exc:
                                record.failure=str(exc)
                                event('current_frame_reselection_rejected',feedback=str(exc))
                                break
                            pending_decision=fresh
                            continue
                        remaining_unknown=planner.recovery_unknown()
                        followup=None
                        if len(remaining_unknown) and record.observations<pregrasp_limit:
                            followup=suggest_views(robot,obs,remaining_unknown,
                                budget=min(2,pregrasp_limit-record.observations),
                                focus_points=target_part,observed_points=map_.surface)
                            event('retry_recovery_view_advice',**followup)
                        history.append({'feedback':record.failure,
                            'coverage_advice':followup,
                            'remaining_unknown_points':len(remaining_unknown),
                            'next_step':('If coverage advice has a view, observe it to certify the exact residual path points; otherwise re-ground away from non-finger target interference. Do not reuse a failed grasp blindly.'
                                if record.observations<pregrasp_limit else
                                'Pre-grasp observation budget is exhausted; preserve the post-close mapping reserve and stop without executing an uncertified grasp.')})
                        # Keep this as one recovery transaction. The new frame
                        # updates geometry only; retry the stored image ray and
                        # candidates instead of silently changing the grasp by
                        # re-grounding from the recovery camera.
                        if followup and followup.get('views') and strategy=='active':
                            residual=vlm.decide(obs,INSTRUCTIONS[task],history,force_observe=True)
                            event('retry_recovery_decision',**residual.model_dump())
                            if residual.action=='observe':
                                sim.move_camera(residual.azimuth,residual.elevation,residual.distance)
                                history.append({'view':obs.view,'decision':residual.model_dump(),
                                    'feedback':'completed residual mapping observation; preserving the original grasp ray'})
                                obs=observe()
                                newly_seen=(resolver.track(obs,target,robot_depths.get(obs.frame_id))
                                    if task!='occluded' else None)
                                if newly_seen is not None:
                                    combined=np.vstack([target,newly_seen])
                                    _,unique=np.unique(np.round(combined/map_.resolution).astype(np.int32),axis=0,return_index=True)
                                    target=combined[unique]
                                target,target_part=reassociate_contacts(obs,decision,original_object,original_part,target,target_part)
                                residual_timer=time.monotonic()
                                planner.rng.bit_generator.state=copy.deepcopy(planning_rng_state)
                                chosen=planner.choose(obs.q,candidates,max_candidates=config.max_candidates,
                                    time_budget=config.planning_seconds,prioritize_gripper_geometry=task=='mug')
                                record.planning_seconds+=time.monotonic()-residual_timer
                                residual_feedback=planner.last_failure if chosen is None else 'planned'
                                (out/f'planning_diagnostic_{obs.frame_id:03d}_residual.json').write_text(json.dumps({
                                    'failure_counts':planner.failure_counts,'closest_invalid':planner.closest_invalid,
                                    'closest_by_failure':planner.closest_by_failure,
                                    'geometry_ranking':getattr(planner,'geometry_ranking',None),
                                    'recovery_unknown_points':len(planner.recovery_unknown()),'checks':planner.checks},indent=2))
                                event('retry_recovery_planning',feedback=residual_feedback,checks=planner.checks,
                                    preserved_grounding_frame=decision.frame_id)
                                if chosen is not None:record.failure=''
                        if chosen is not None:
                            pass
                        elif record.observations<pregrasp_limit:continue
                        else:
                            event('observation_budget_reserved',observations=record.observations,
                                reserved_after_close=reserve_after_close)
                            break
                    record.failure=""
                else:
                    obs=observe(); continue
            grasp,transit,approach=chosen
            record.attempts+=1
            record.failure="";sim.episode_attempt=record.attempts
            execute(transit,-1,"pregrasp"); execute(approach,-1,"approach")
            sim.phase="close"
            for _ in range(25): sim.step(approach[-1],1)
            planner.gripper=sim.gripper_q
            p,R=robot.pose(sim.q)
            supported_grasp_p=p.copy();supported_grasp_R=R.copy()
            source_support_z=map_.table_z
            preclose_target=target.copy()
            map_.certify_robot_state(robot.points(sim.q,sim.gripper_q)[0],
                robot.volumes(sim.q,sim.gripper_q))
            # The target has moved with the closing fingers. Remove its old
            # observed endpoints before probing the lift corridor; static scene
            # endpoints remain.  Run the *actual* IK / interpolated robot check
            # once so active view selection receives the precise rejected
            # samples, rather than only a nominal Cartesian sweep estimate.
            map_.remove_target_endpoints(preclose_target)
            planner.attached=None;planner.gripper=sim.gripper_q
            preview_p,preview_R=robot.pose(sim.q);preview_q=sim.q.copy()
            preview_path=[preview_q];preview_ik=True
            for dz in np.linspace(.01,.13,14):
                preview=robot.ik(preview_p+[0,0,dz],preview_R,preview_q)
                if preview is None:
                    preview_ik=False;break
                preview_q=preview;preview_path.append(preview)
            # Sample the same 0.025 rad interpolation resolution used by the
            # execution checker.  Looking only at Cartesian waypoints misses
            # thin visibility gaps traversed between two valid IK solutions.
            def classify_preview_sweep():
                unseen_sets=[]
                for a,b in zip(preview_path[:-1],preview_path[1:]):
                    n=max(2,int(np.ceil(np.max(np.abs(b-a))/.025))+1)
                    for sample_q in np.linspace(a,b,n):
                        sample_points=robot.points(sample_q,sim.gripper_q)[0]
                        unseen=sample_points[map_.classify(sample_points)==0]
                        if len(unseen):unseen_sets.append(unseen)
                if not unseen_sets:return np.empty((0,3))
                points=np.vstack(unseen_sets)
                _,unique=np.unique(np.round(points/map_.resolution).astype(np.int32),axis=0,return_index=True)
                return points[unique]
            preview_unknown=classify_preview_sweep()
            preview_ok=preview_ik and not len(preview_unknown)
            event('postgrasp_lift_preview',feasible=preview_ok,
                feedback='planned' if preview_ok else ('IK' if not preview_ik else 'unknown space'),
                rejected_unknown_points=len(preview_unknown))
            robot.set_q(sim.q,sim.gripper_q)
            # Closing can move the object before lift. A holed mug may use one
            # additional frame; solid-object tasks remain capped at two. The independent evaluator still
            # verifies success if the observation budget is fully consumed.
            post_index=0;post_association=False
            post_limit=phase_budget.postclose_limit
            max_post_views=reserve_after_close
            while record.observations<post_limit and post_index<max_post_views:
                sweep=np.vstack([target+np.array([0.,0.,dz]) for dz in np.linspace(0,.13,14)])
                sweep_unknown=sweep[map_.classify(sweep)==0]
                leading=leading_shell(sweep_unknown,map_.resolution)
                # Weight the leading boundary: it enters new space first. The
                # rest of the sweep remains represented once, but cannot drown
                # out the thin top shell in a tall handled object.
                advice_points=np.vstack([sweep_unknown,np.repeat(leading,4,axis=0)]) if len(leading) else sweep_unknown
                # Also predict the arm's vertical-lift sweep using only known
                # kinematics. Payload-only view selection can leave a wrist or
                # elbow surface just behind the currently observed robot.
                lift_robot=[];preview_q=sim.q.copy()
                for dz in np.linspace(.01,.13,14):
                    preview=robot.ik(p+[0,0,dz],R,preview_q)
                    if preview is None:break
                    lift_robot.append(robot.points(preview,sim.gripper_q)[0]);preview_q=preview
                if lift_robot:
                    robot_sweep=np.vstack(lift_robot)
                    robot_unknown=robot_sweep[map_.classify(robot_sweep)==0]
                    if len(robot_unknown):
                        _,robot_unique=np.unique(np.round(robot_unknown/map_.resolution).astype(np.int32),axis=0,return_index=True)
                        robot_unknown=robot_unknown[robot_unique]
                else:robot_unknown=np.empty((0,3))
                if (post_index and not len(sweep_unknown) and not len(robot_unknown)
                        and (task!='occluded' or post_association)):break
                current_obs=replace(obs,q=sim.q.copy(),gripper_q=sim.gripper_q.copy())
                # Whole-arm safety is a hard priority: a high payload point
                # count must not sacrifice even one unseen wrist sample. Use
                # payload leading surfaces and target visibility only to break
                # ties among views that cover the robot sweep.
                # The exact samples rejected by the interpolated IK path take
                # precedence on the first post-grasp view.  Later iterations
                # fall back to the recomputed kinematic sweep after the map has
                # changed.
                exact_unknown=preview_unknown
                required_points=exact_unknown if len(exact_unknown) else robot_unknown
                # First satisfy every coverable whole-arm sample; then rank
                # those views by the carried object's leading lift sweep. This
                # avoids spending the only post-close frame on the arm while
                # leaving the payload's millimetre-scale upper shadow unknown.
                coverage_points=advice_points if len(advice_points) else required_points
                focus=np.vstack([target,leading]) if len(leading) else target
                # When any robot sweep point is unseen, do not let attractive
                # target framing break visibility ties.  The target is already
                # localized; this view exists first and foremost to certify the
                # arm corridor. Payload focus is used only once the arm is fully
                # observed.
                focus_for_advice=None if len(required_points) else focus
                # Retiring a moving payload from the static collision map must
                # not make the camera model see through its observed surfaces.
                # This local union changes visibility ranking only; it never
                # reintroduces static obstacles or certifies UNKNOWN as free.
                advice_occluders=np.vstack([map_.surface,target])
                payload_advice=suggest_views(robot,current_obs,coverage_points,budget=1,
                    focus_points=focus_for_advice,observed_points=advice_occluders,
                    required_points=required_points if len(required_points) else None,
                    semantic_focus=target if task=='occluded' else None)
                payload_advice['occlusion_evidence']={
                    'basis':'static_surface_plus_observed_payload_reference',
                    'static_surface_points':len(map_.surface),'payload_reference_points':len(target),
                    'scope':'Visibility prediction only; observed references do not certify current visibility.',
                    'visibility_prediction_tolerance_unchanged':True}
                history.append({'coverage_advice':payload_advice,
                    'scene_changed_since_observation':post_index==0,'feedback':
                    'The gripper has closed with observed contact. Choose one mapping view for the carried-object lift sweep; do not ground again.'})
                event('postgrasp_view_advice',**payload_advice)
                if strategy == 'active':
                    post=vlm.decide(obs,INSTRUCTIONS[task],history,force_observe=True)
                    event('postgrasp_decision',**post.model_dump())
                    selected=(post.azimuth,post.elevation,post.distance)
                else:
                    selected=None
                selected=tabletop_view(strategy,initial_view,rng,selected)
                sim.move_camera(*selected)
                history.append({'view':obs.view,'selected_view':selected,
                    'feedback':'completed one post-grasp mapping observation'})
                obs=observe(after_close=True)
                removed_sets=[]
                if task=='occluded':
                    association=contact_annotation(obs,decision,original_object,original_part,
                        'post-close object and permitted-part verification before lifting',source_supported=True)
                    # Camera-only observations do not advance physics. Keep a
                    # verified post-close association through later mapping
                    # views that cannot see the part; the next physical action
                    # ends this evidence epoch and requires its own checks.
                    post_association=post_association or association is not None
                    if association is not None:
                        target=np.unique(np.vstack([target,association.object_points]),axis=0)
                        target_part=np.unique(np.vstack([target_part,association.part_points]),axis=0)
                        map_.set_target(target_part)
                        removed_sets=[association.object_points]
                    current_target=None
                else:
                    current_target=resolver.track(obs,target,robot_depths.get(obs.frame_id),support_z=map_.table_z)
                if current_target is not None and len(current_target)>=20:
                    # Different views expose different surfaces: a median shift
                    # is not object motion. Reassociate the contact neighborhood
                    # directly in the shared calibrated world frame.
                    center=grasp.grasp_center
                    current_part=current_target[np.linalg.norm(current_target-center,axis=1)<.04]
                    if len(current_part):target_part=current_part
                    # The first post-close RGB-D cloud supersedes the stale
                    # pre-close payload. Subsequent views are already calibrated
                    # in the same world frame and are fused without a second
                    # viewpoint-biased median translation.
                    target=current_target if post_index==0 else np.vstack([target,current_target])
                    _,target_unique=np.unique(np.round(target/map_.resolution).astype(np.int32),axis=0,return_index=True)
                    target=target[target_unique]
                    map_.set_target(target_part)
                # Occluded pick/place retires only the fresh explicit semantic
                # association. A loose tracker over historical views can merge
                # a nearby independent object and erase its collision evidence.
                if task!='occluded':
                    for prior_obs in map_.views:
                        seen=resolver.track(prior_obs,target,robot_depths.get(prior_obs.frame_id),support_z=map_.table_z)
                        if seen is not None:removed_sets.append(seen)
                if removed_sets:
                    stale=np.vstack(removed_sets)
                    map_.remove_target_endpoints(stale)
                    event('postgrasp_perception',associated_surface_points=len(stale),
                        current_payload_points=len(target),leading_unknown_points=len(leading),
                        robot_lift_unknown_points=len(robot_unknown))
                # Close the loop: score a second view from what remains unknown
                # after the first *actual* RGB-D integration.
                preview_unknown=classify_preview_sweep()
                event('postgrasp_residual',rejected_unknown_points=len(preview_unknown),
                    postgrasp_view_index=post_index+1)
                post_index+=1
            if task=='occluded' and not post_association:
                raise RuntimeError('execution: post-close object/part association unverified; lift blocked')
            p,R=robot.pose(sim.q)
            planner.attached=(target-p)@R
            planner.observed_payload_local=None
            attached_part=(target_part-p)@R
            np.save(out/'payload_source_points.npy',target)
            np.savez(out/'payload_attachment_pose.npz',position=p,rotation=R)
            map_.set_payload_source(target)
            map_.remove_target_endpoints(target)
            planner.begin_payload_lift()
            # Lift diagnostics must describe this phase, not the rejected
            # pregrasp branches retained from the preceding candidate search.
            planner.closest_by_failure={};planner.closest_invalid=None;planner.payload_invalid=None
            # Build a lift path from robot proprioception and observed grasp geometry.
            lift=[sim.q];lift_timer=time.monotonic()
            planner.allow_payload_support=True
            try:
                for dz in np.linspace(.01,.13,14):
                    q=planner.feasible_ik(p+[0,0,dz],R,lift[-1],contact=True,connected=True)
                    if q is None:
                        (out/'lift_planning_diagnostic.json').write_text(json.dumps({
                            'failure':planner.last_failure,'payload_invalid':planner.payload_invalid,
                            'closest_by_failure':planner.closest_by_failure},indent=2))
                        raise RuntimeError("planning: lift "+planner.last_failure)
                    lift.append(q)
            finally:
                planner.allow_payload_support=False;planner.end_payload_lift()
                record.planning_seconds+=time.monotonic()-lift_timer
            execute(np.array(lift),1,"lift")
            sim.phase="hold"
            for _ in range(45): sim.step(lift[-1],1)
            tracked_part=None;post_lift_observations_used=0
            if phase_budget.postlift_observations and record.observations>=config.max_observations:
                raise RuntimeError('execution: reserved post-lift verification observation unavailable')
            if record.observations<config.max_observations:
                obs=observe(after_close=True)
                post_lift_observations_used+=1
                if task=='occluded':
                    held_p,held_R=robot.pose(sim.q)
                    held_object=planner.attached@held_R.T+held_p
                    held_part=attached_part@held_R.T+held_p
                    association=contact_annotation(obs,decision,held_object,held_part,
                        'post-lift object and permitted-part verification')
                    if (association is None and strategy!='fixed'
                            and post_lift_observations_used<phase_budget.postlift_observations
                            and record.observations<config.max_observations):
                        # A failed polygon is not proof of grasp loss, nor
                        # permission to move. Spend the one remaining reserve
                        # on a different actual camera view of this unchanged
                        # held-pose hypothesis, without moving the robot.
                        try:
                            advice_timer=time.monotonic()
                            robot.set_q(sim.q,sim.gripper_q)
                            current_obs=replace(obs,q=sim.q.copy(),gripper_q=sim.gripper_q.copy())
                            try:
                                candidates=suggest_views(robot,current_obs,np.empty((0,3)),budget=2,
                                    focus_points=held_part,semantic_focus=held_object,
                                    observed_points=map_.surface)
                            finally:record.planning_seconds+=time.monotonic()-advice_timer
                            views=[v for v in candidates.get('views',[]) if different_tabletop_view(v,obs.view)]
                            held_advice={'purpose':'held_object_verification','views':views,
                                'max_observations_to_acquire':1,'views_are_alternatives':True,
                                'predicted_semantic_focus':[s for s in candidates.get('predicted_semantic_focus',[])
                                    if different_tabletop_view(s['view'],obs.view)],
                                'reference_kind':'fixed_observed_attachment_at_current_robot_pose',
                                'scope':'Visibility hypotheses only. No UNKNOWN path samples, free-space certificate, new grasp ray, or contact permission.'}
                            history.append({'coverage_advice':held_advice,'held_verification':{
                                'frame_id':obs.frame_id,'camera_id':obs.camera_id,
                                'strict_association_verified':False,'remaining_real_observations':1,
                                'remaining_transfer_mapping_observations_after_this_view':0},'feedback':
                                'The robot is stationary after lifting, but this image did not strictly verify the held object and permitted part. Choose ONE DIFFERENT real camera view that makes that same held object and part identifiable. The latest advice is specifically for held-object visibility, not an UNKNOWN corridor or a new grasp. Do not re-ground a grasp or change the original ray. The fixed attachment projection is only a hypothesis. After the new image, explicit object and part annotations must pass all original strict checks and measured lift verification. Using this last reserved view leaves no transfer mapping observation.'})
                            event('held_verification_view_advice',frame_id=obs.frame_id,**held_advice)
                            if strategy=='active':
                                post=vlm.decide(obs,INSTRUCTIONS[task],history,force_observe=True)
                                event('held_verification_view_decision',**post.model_dump())
                                if post.action!='observe' or post.frame_id!=obs.frame_id or post.camera_id!=obs.camera_id:
                                    raise ValueError('held verification requires a current-frame observe decision')
                                selected=(post.azimuth,post.elevation,post.distance)
                            else:selected=None
                            selected=tabletop_view(strategy,initial_view,rng,selected)
                            if not different_tabletop_view(selected,obs.view):
                                raise ValueError('held verification recovery repeated the current view')
                            sim.move_camera(*selected)
                            obs=observe(after_close=True);post_lift_observations_used+=1
                            association=contact_annotation(obs,decision,held_object,held_part,
                                'post-lift held-object verification from the last reserved new view')
                            event('held_verification_recovery',frame_id=obs.frame_id,
                                association_verified=association is not None,
                                post_lift_observations_used=post_lift_observations_used,
                                remaining_transfer_observations=phase_budget.postlift_observations-post_lift_observations_used)
                        except (ValueError,RuntimeError) as exc:
                            event('held_verification_recovery_rejected',frame_id=obs.frame_id,reason=str(exc))
                            association=None
                    tracked=None if association is None else association.object_points
                    tracked_part=None if association is None else association.part_points
                else:
                    tracked=resolver.track(obs,target+[0,0,.13],robot_depths.get(obs.frame_id),support_z=map_.table_z)
                lifted=tracked is not None and float(np.median(tracked[:,2]))>float(np.median(target[:,2]))+.07 and abs(sim.gripper_q[0]-sim.gripper_q[1])>.006
                event("visual_verification",lifted=bool(lifted))
            else:
                # The observation budget may be spent resolving the strict
                # carried-object / whole-arm sweep. Do not feed privileged task
                # truth back to the agent; the independent evaluator scores the
                # completed 2 s hold after the episode.
                tracked=None;lifted=abs(sim.gripper_q[0]-sim.gripper_q[1])>.006
                event("verification_deferred_to_independent_evaluator")
            if lifted and record.attempts==1: record.first_grasp_success=True
            if not lifted:
                record.failure="execution: visual lift verification failed"
                history.append({"feedback":record.failure})
                # Release only after a planned return to the grasp height.
                execute(np.asarray(lift)[::-1],1,"lower")
                sim.phase="release"
                for _ in range(20): sim.step(sim.q,-1)
                planner.attached=None; planner.gripper=None; map_.clear_dynamic()
                if record.observations<pregrasp_limit: obs=observe(); continue
                break
            # The world is static except for the grasped target. Remove its old
            # endpoints, then exclude the observed held target from the obstacle map.
            map_.remove_target_endpoints(target)
            if tracked is not None:
                if task=='occluded':
                    # Strict held semantics may expose previously unseen sides.
                    # Keep them in every future payload collision check before
                    # retiring their measured endpoints from the static map.
                    held_p,held_R=robot.pose(sim.q)
                    planner.add_observed_payload(tracked,held_p,held_R)
                map_.remove_target_endpoints(tracked)
                map_.set_target(tracked_part if task=='occluded' else tracked)
            held_p,held_R=robot.pose(sim.q)
            held_estimate=planner.attached@held_R.T+held_p
            # A predicted attachment pose is not observed object membership.
            # Keep actual occupied neighbors even near the expected held body.
            if task!='occluded':map_.remove_target_endpoints(held_estimate)
            collision_payload=(planner.payload_collision_local()@held_R.T+held_p
                if task=='occluded' else held_estimate)
            map_.set_payload_source(collision_payload)
            if task!="occluded": break
            if decision.place_point is None: raise RuntimeError("grounding: missing destination")
            # place_point refers to the pre-motion frame; recover that calibrated observation.
            old=np.load(out/"observations"/f"frame_{decision.frame_id:03d}.npz")
            h,w=old['depth'].shape
            uv=np.rint(np.array(decision.place_point)*[w-1,h-1]).astype(int)
            destination=unproject(uv,old["depth"][uv[1],uv[0]],old["K"],old["T_wc"])
            current_p,current_R=robot.pose(sim.q)
            # This tabletop task starts with the target supported on this same
            # observed horizontal table. Visible top/side points do not locate
            # its hidden physical bottom. Preserve the pre-lift supported pose
            # and release with a gap, while centering the observed object
            # reference (rather than the TCP) on the grounded destination.
            placement=tabletop_placement(destination_xy=destination[:2],
                observed_payload_local=planner.attached,
                source_tcp_position=supported_grasp_p,source_rotation=supported_grasp_R,
                current_tcp_position=current_p,current_rotation=current_R,
                source_support_z=source_support_z,destination_support_z=map_.table_z,
                source_supported=True)
            event('placement_reference',**placement.diagnostic())
            (out/'placement_reference.json').write_text(json.dumps(placement.diagnostic(),indent=2))
            destination=placement.transfer_position;current_R=placement.rotation
            transfer_index=0;transfer_rng_state=None
            while True:
                # Reuse the search samples after new real image evidence. A
                # camera-policy RNG draw must not turn recovery into a lucky
                # unrelated RRT retry on an otherwise unchanged map.
                if transfer_rng_state is not None:
                    rng.bit_generator.state=copy.deepcopy(transfer_rng_state)
                transfer_timer=time.monotonic()
                try:
                    attempt=plan_transfer_attempt(planner,sim.q,destination,current_R,
                        time_budget=config.planning_seconds)
                except Exception as exc:
                    (out/f'transfer_planning_diagnostic_{transfer_index:03d}.json').write_text(json.dumps({
                        'phase':'transfer','attempt':transfer_index,'frame_id':obs.frame_id,
                        'failure':str(exc),'elapsed_seconds':time.monotonic()-transfer_timer,
                        'observation_recovery_eligible':False},indent=2))
                    raise
                finally:
                    record.planning_seconds+=time.monotonic()-transfer_timer
                if transfer_rng_state is None:transfer_rng_state=attempt.rng_state
                transfer_report={**attempt.report,'attempt':transfer_index,'frame_id':obs.frame_id,
                    'rng_restored':transfer_index>0,'observations_used':record.observations,
                    'post_lift_observations_used':post_lift_observations_used,
                    'post_lift_observations_reserved':phase_budget.postlift_observations,
                    'attachment_policy':'fixed original semantic/contact/placement reference; strictly associated new surfaces enlarge collision-only payload',
                    'collision_payload_points':len(planner.payload_collision_local())}
                (out/f'transfer_planning_diagnostic_{transfer_index:03d}.json').write_text(json.dumps(transfer_report,indent=2))
                event('transfer_planning',attempt=transfer_index,frame_id=obs.frame_id,
                    path_found=attempt.path is not None,failure=attempt.report['failure'],
                    elapsed_seconds=attempt.report['elapsed_seconds'],
                    rejected_unknown_points=len(attempt.unknown_points),rng_restored=transfer_index>0)
                transfer=attempt.path
                if transfer is not None:break
                if not (phase_budget.postlift_observations>post_lift_observations_used
                        and record.observations<config.max_observations
                        and attempt.report['observation_recovery_eligible']):
                    raise RuntimeError("planning: transfer "+planner.last_failure)
                # Geometry search leaves the scratch robot at a rejected
                # sample. View planning starts at the real stationary held
                # pose and uses only actual UNKNOWN samples from that search.
                robot.set_q(sim.q,planner.gripper)
                held_p,held_R=robot.pose(sim.q)
                held_object=planner.attached@held_R.T+held_p
                held_part=attached_part@held_R.T+held_p
                current_obs=replace(obs,q=sim.q.copy(),gripper_q=sim.gripper_q.copy())
                transfer_advice=suggest_views(robot,current_obs,attempt.unknown_points,budget=1,
                    required_points=attempt.unknown_points,focus_points=held_part,
                    semantic_focus=held_object,observed_points=map_.surface)
                history.append({'coverage_advice':transfer_advice,'feedback':
                    'Transfer was rejected by actual UNKNOWN space. Choose one real mapping view covering these rejected samples and the visible held object. This view does not approve motion; the same object and permitted part must be verified again before replanning.'})
                event('transfer_view_advice',**transfer_advice)
                if strategy=='active':
                    post=vlm.decide(obs,INSTRUCTIONS[task],history,force_observe=True)
                    event('transfer_recovery_decision',**post.model_dump())
                    if post.action!='observe' or post.frame_id!=obs.frame_id or post.camera_id!=obs.camera_id:
                        raise RuntimeError('grounding: transfer recovery requires a current-frame observation decision')
                    selected=(post.azimuth,post.elevation,post.distance)
                else:selected=None
                selected=tabletop_view(strategy,initial_view,rng,selected)
                sim.move_camera(*selected)
                obs=observe(after_close=True);post_lift_observations_used+=1
                association=contact_annotation(obs,decision,held_object,held_part,
                    'post-lift transfer recovery object and permitted-part verification')
                if association is None:
                    raise RuntimeError('execution: transfer recovery held-object association unverified; transfer blocked')
                # Represent every retired new surface in the held collision
                # envelope. The original semantic/placement/contact reference
                # remains fixed; the extra surface can only add refusals.
                planner.add_observed_payload(association.object_points,held_p,held_R)
                map_.remove_target_endpoints(association.object_points)
                map_.set_target(association.part_points)
                map_.set_payload_source(planner.payload_collision_local()@held_R.T+held_p)
                event('transfer_recovery_perception',frame_id=obs.frame_id,
                    associated_surface_points=len(association.object_points),
                    permitted_part_points=len(association.part_points),
                    post_lift_observations_used=post_lift_observations_used)
                transfer_index+=1
            execute(transfer,1,"transfer")
            final_z=placement.release_position[2]
            lower=[sim.q];lower_timer=time.monotonic()
            planner.allow_payload_support=True
            try:
                for z in np.linspace(destination[2],final_z,20)[1:]:
                    q=robot.ik(np.r_[destination[:2],z],current_R,lower[-1])
                    if q is None or not planner.edge(lower[-1],q,contact=True): raise RuntimeError("planning: lower "+planner.last_failure)
                    lower.append(q)
            finally:
                planner.allow_payload_support=False
                record.planning_seconds+=time.monotonic()-lower_timer
            execute(np.array(lower),1,"lower")
            sim.phase="release"
            for _ in range(60): sim.step(lower[-1],-1)
            break
        if not record.failure and not record.attempts: record.failure="observation budget exhausted"
    except KeyboardInterrupt:
        record.failure="interrupted during development"
        event("interrupted")
        raise
    except Exception as exc:
        record.failure=str(exc)
        (out/"error.txt").write_text(traceback.format_exc())
        event("error",message=str(exc))
    finally:
        if vlm:
            record.vlm_calls=vlm.calls; record.inference_seconds=vlm.seconds; vlm.client.close()
        if 'resolver' in locals():
            record.segmentation_calls=resolver.calls;record.perception_seconds=resolver.seconds
        record.elapsed_seconds=time.monotonic()-start
        if sim:
            if sim.evaluator:
                score=sim.evaluator.result()
                record.success=score["success"]; record.collision=score["collision"]
                record.first_grasp_success=score["first_grasp_success"]
                record.collision_free_success=score['collision_free_success']
                (out/"evaluation_truth.json").write_text(json.dumps(score,indent=2))
            if record.success: record.failure=""
            elif not record.failure: record.failure="execution: evaluator success criterion not met"
            save_episode(out,sim,record,events)
            if 'robot' in locals(): robot.close()
            sim.close()
        else: (out/"result.json").write_text(json.dumps(asdict(record),indent=2))
    return asdict(record)
