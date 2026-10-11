"""Exercise bounded annotation-only recovery through the real runner loop."""
from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from test_contact_reassociation import annotations,scene
from view2grasp.grounding import GroundingResolver
from view2grasp.geometry import unproject
from view2grasp.types import GraspCandidate


@pytest.mark.parametrize('annotation_outcome,post_close,post_lift,crop_mode',[
    ('accepted',False,False,'default'),('missing_part',False,False,'default'),('api_failure',False,False,'default'),
    ('accepted',True,False,'default'),('missing_part',True,False,'default'),('api_failure',True,False,'default'),
    ('overbroad_neighbor',True,False,'default'),('accepted',True,True,'default'),
    ('api_failure',True,True,'default'),('overbroad_neighbor',True,True,'default'),
    ('accepted',False,False,'cropped'),('accepted',False,False,'fallback'),
    ('repair_then_accept',False,False,'cropped'),
    ('progress_then_accept',False,False,'cropped'),('progress_then_reject',False,False,'cropped'),
    ('topology_then_accept',False,False,'cropped'),('topology_then_reject',False,False,'cropped'),
    ('reference_then_accept',False,False,'cropped'),('reference_then_reject',False,False,'cropped'),
    ('accepted',False,False,'multiple_mapping_views'),
])
def test_mapping_reassociation_preserves_ray_and_never_moves_for_annotation(
        tmp_path,monkeypatch,annotation_outcome,post_close,post_lift,crop_mode):
    from view2grasp import runner,recording
    multiple_mapping_views=crop_mode=='multiple_mapping_views'
    sample,xyz,mask,part_reference=scene()
    padding=200 if crop_mode=='cropped' else 0
    if padding:
        depth=np.pad(sample.depth,padding,constant_values=2.)
        K=sample.K.copy();K[:2,2]+=padding
        sample=replace(sample,rgb=np.pad(sample.rgb,((padding,padding),(padding,padding),(0,0))),depth=depth,K=K)
        mask=np.pad(mask,padding)
        v,u=np.mgrid[:depth.shape[0],:depth.shape[1]]
        xyz=unproject(np.stack([u,v],-1),depth,K,sample.T_wc)
    original_part=np.zeros_like(mask);original_part[10+padding:14+padding,8+padding:10+padding]=True
    part_reference=xyz[original_part]
    topology_repair=annotation_outcome in ('topology_then_accept','topology_then_reject')
    original_object=part_reference.copy() if multiple_mapping_views or topology_repair else xyz[mask]
    evidence=SimpleNamespace(cameras=[],observations=[],targets=[],candidate_lists=[],vlm_calls=[],events=[],
        removal_requests=[],remaining_obstacles=[],executed=[],lift_checks=0,measured_objects=[],annotation_shapes=[],association_shapes=[],
        association_frames=[],association_references=[],planning_association_frames=[],annotation_replies={})
    candidate=GraspCandidate(np.zeros(3),np.eye(3),.02,np.array([0.,0.,1.]),1.,0.,
        surface_point=part_reference[0],grasp_center=part_reference[0])
    original_ray=candidate.approach.copy()

    class Sim:
        def __init__(self,*args):
            self.q=np.zeros(7);self.gripper_q=np.array([.04,-.04]);self.frame=0
            self.robot_xml='';self.joint_names=[];self.eef_name='';self.phase='observe'
            evidence.sim=self
        def observe(self):
            self.frame+=1;evidence.observations.append(self.frame)
            depth=sample.depth.copy()
            if self.phase=='hold':depth[mask]+=self.q[0]
            return replace(sample,frame_id=self.frame,depth=depth,q=self.q.copy(),gripper_q=self.gripper_q.copy())
        def move_camera(self,*view):evidence.cameras.append(view)
        def validate_initial_proxies(self,*args):pass
        def step(self,q,grip):
            assert post_close,'this episode must not execute an uncertified grasp'
            self.q=np.asarray(q).copy();evidence.executed.append(self.phase)
        def close(self):pass

    class Robot:
        def __init__(self,*args):pass
        def points(self,*args):return np.zeros((1,3)),None
        def volumes(self,*args):return ()
        def visual_volumes(self,*args):return ()
        def depth(self,obs):return np.full_like(obs.depth,10.)
        def pose(self,q=None):
            q=evidence.sim.q if q is None else q
            return np.array([0.,0.,q[0]]),np.eye(3)
        def grasp_center_offset(self):return np.zeros(3)
        def ik(self,p,R,q):return np.r_[p[2],q[1:]]
        def set_q(self,*args):pass
        def close(self):pass

    neighbor=xyz[10+padding:14+padding,12+padding:16+padding].reshape(-1,3)
    class Map(runner.DepthMap):
        def __init__(self,resolution):
            super().__init__(resolution,workspace_bounds=None)
            self.surface=np.vstack([original_object,neighbor]);self.tree=cKDTree(self.surface)
            self.table_z=.95;self.views=[]
        def certify_home(self,*args):pass
        def certify_robot_state(self,*args):pass
        def integrate(self,obs,*args):
            self.views.append(obs)
            v,u=np.where(mask);measured=unproject(np.c_[u,v],obs.depth[v,u],obs.K,obs.T_wc)
            evidence.measured_objects.append(measured)
            self.surface=np.vstack([measured,neighbor]);self.tree=cKDTree(self.surface)
            self._refresh_non_target()
        def set_table(self,z):self.table_z=z
        def classify(self,points):return np.full(len(points),int(post_lift and len(self.views)>=3),np.uint8)
        def set_target(self,points):
            evidence.targets.append(points.copy());super().set_target(points)
        def remove_target_endpoints(self,points,**kwargs):
            evidence.removal_requests.append(points.copy())
            super().remove_target_endpoints(points,**kwargs)
            evidence.remaining_obstacles.append(self.surface.copy())

    class Plan:
        add_observed_payload=runner.Planner.add_observed_payload
        payload_collision_local=runner.Planner.payload_collision_local
        def __init__(self,robot,map_,rng):
            self.rng=rng;self.calls=0;self.checks=0;self.last_failure='unknown space'
            self.failure_counts={};self.closest_invalid=None;self.closest_by_failure={}
            self.best_unknown=original_object[:1];self.payload_invalid=None
        def recovery_unknown(self):return original_object[:1]
        def choose(self,q,candidates,**kwargs):
            evidence.candidate_lists.append(candidates)
            evidence.planning_association_frames.append(evidence.association_frames.copy())
            self.calls+=1;self.checks+=1
            if post_close and self.calls==2:return candidates[0],np.array([q,q]),np.array([q,q])
            return None
        def begin_payload_lift(self):pass
        def end_payload_lift(self):pass
        def feasible_ik(self,p,R,q,**kwargs):
            evidence.lift_checks+=1;self.last_failure='deliberate lift rejection'
            return np.r_[p[2],q[1:]] if post_lift else None

    class Decisions:
        def __init__(self,*args,**kwargs):
            self.calls=0;self.seconds=0.;self.client=SimpleNamespace(close=lambda:None)
        def decide(self,obs,instruction,history,**kwargs):
            self.calls+=1;evidence.vlm_calls.append((obs.frame_id,kwargs,json.loads(json.dumps(history))))
            assert self.calls<=(9 if post_close else 7 if multiple_mapping_views else 5),'annotation must not create an unbounded repair loop'
            if self.calls==1:
                evidence.global_history=history
                return annotations(obs,mask,original_part).model_copy(update={'action':'grasp'})
            if 'contact_reassociation' not in history[-1]:
                assert history is evidence.global_history
                return annotations(obs,mask).model_copy(update={'groundings':[]})
            evidence.annotation_shapes.append(obs.depth.shape)
            assert kwargs['observe_purpose']=='annotation'
            # Exercise the actual runner with old grasp pixels, mapping advice,
            # and (in multi-frame cases) earlier annotations in global history.
            # Only the current acquired image's transaction goes to the VLM.
            assert len(history)==1 and set(history[0])=={'contact_reassociation'}
            assert history is not evidence.global_history and len(evidence.global_history)>1
            assert any('coverage_advice' in item for item in evidence.global_history)
            assert any(item.get('decision',{}).get('frame_id')==1 for item in evidence.global_history)
            transaction=history[0]['contact_reassociation']
            assert transaction['frame_id']==obs.frame_id and transaction['camera_id']==obs.camera_id
            assert transaction is not evidence.global_history[-1]['contact_reassociation']
            assert transaction['image_context'] is not evidence.global_history[-1]['contact_reassociation']['image_context']
            if 'previous_annotation' in transaction:
                assert transaction['previous_annotation']==evidence.annotation_replies[obs.frame_id]
                assert transaction['previous_annotation']['frame_id']==obs.frame_id
                assert transaction['geometry_rejection']['frame_id']==obs.frame_id
                assert transaction['geometry_rejection']['camera_id']==obs.camera_id
            else:
                assert 'geometry_rejection' not in transaction
            # A client-side mutation must not leak back into global planning or
            # the next correction's copied source transaction.
            assert 'client_mutation' not in transaction
            transaction['client_mutation']='annotation-local-only'
            image_context=history[-1]['contact_reassociation']['image_context']
            assert kwargs['derived_image_context']==(None if crop_mode=='fallback' else image_context)
            diagnostic=kwargs.get('annotation_diagnostic')
            repair_feedback=history[-1]['contact_reassociation'].get('geometry_rejection')
            assert (diagnostic is not None)==(crop_mode!='fallback' or repair_feedback is not None)
            if diagnostic is not None:
                # Geometry and returned coordinates still use the acquired
                # observation, independently of the extra diagnostic panel.
                assert diagnostic.rgb.shape!=obs.rgb.shape
                assert obs.rgb.shape[:2]==obs.depth.shape
                if repair_feedback is None:assert not diagnostic.rejected_pixels
            outcome=('accepted' if (post_close and obs.frame_id==2) or
                (post_lift and obs.frame_id<4) else annotation_outcome)
            if outcome=='repair_then_accept':
                repair_context=history[-1]['contact_reassociation'].get('geometry_rejection')
                if repair_context is None:outcome='overbroad_neighbor'
                else:
                    outcome='accepted'
                    assert repair_context['coordinate_space']=='supplied_crop_normalized_pixel_centers'
                    x0,y0,x1,y1=image_context['bounds_xyxy']
                    for rejected in repair_context['rejected_samples']:
                        source=np.asarray(rejected['source_pixel'])
                        expected=(source-[x0,y0])/[x1-x0-1,y1-y0-1]
                        assert np.allclose(rejected['normalized'],expected)
                    assert 'previous_annotation' in history[-1]['contact_reassociation']
            if outcome=='api_failure':raise RuntimeError('VLM service unavailable in mock')
            proposed_mask=mask.copy()
            if outcome=='overbroad_neighbor':proposed_mask[10+padding:14+padding,12+padding:16+padding]=True
            if outcome in ('progress_then_accept','progress_then_reject'):
                if self.calls==3:
                    proposed_mask[:]=False
                    proposed_mask[10+padding:14+padding,12+padding:16+padding]=True
                elif self.calls==4 or outcome=='progress_then_reject':
                    proposed_mask[10+padding:14+padding,12+padding:16+padding]=True
            if outcome in ('topology_then_accept','topology_then_reject'):
                if self.calls==3:
                    proposed_mask[10+padding:14+padding,12+padding:16+padding]=True
                elif self.calls==4 or outcome=='topology_then_reject':
                    proposed_mask[:,10+padding]=False
            if outcome in ('reference_then_accept','reference_then_reject'):
                if self.calls==3:
                    proposed_mask[6+padding:8+padding,8+padding:24+padding]=True
                elif self.calls==4 or outcome=='reference_then_reject':
                    proposed_mask[10+padding]=False
                    proposed_mask[8+padding,8+padding]=True
            proposed_part=original_part if post_close else None
            if 'bounds_xyxy' in image_context:
                x0,y0,x1,y1=image_context['bounds_xyxy']
                proposed_mask=proposed_mask[y0:y1,x0:x1]
                if proposed_part is not None:proposed_part=proposed_part[y0:y1,x0:x1]
                assert np.array_equal(obs.depth,sample.depth[y0:y1,x0:x1]) or post_close
                assert np.array_equal(obs.K[:2,2],sample.K[:2,2]-[x0,y0])
            result=annotations(obs,proposed_mask,proposed_part)
            if crop_mode=='cropped':
                result=result.model_copy(update={'point':(.5,.5)})
                evidence.expected_source_pixel=np.array([x0+64,y0+64])
            if outcome=='missing_part':result=result.model_copy(update={'groundings':result.groundings[:1]})
            evidence.annotation_replies[obs.frame_id]=result.model_dump()
            return result

    resolver=GroundingResolver()
    original_reassociate=resolver.reassociate_contact
    def reassociate(obs,decision,*args,**kwargs):
        evidence.association_shapes.append(obs.depth.shape)
        evidence.association_frames.append(obs.frame_id)
        evidence.association_references.append((args[0].copy(),args[1].copy()))
        assert obs.depth.shape==sample.depth.shape
        assert all(tuple(g.mask_size)==sample.depth.shape for g in decision.groundings)
        assert args[-1].shape==sample.depth.shape # Original known-robot depth, not the cropped grid.
        if crop_mode=='cropped':
            assert np.array_equal(np.rint(np.asarray(decision.point)*(np.array(sample.depth.shape[::-1])-1)),
                evidence.expected_source_pixel)
        return original_reassociate(obs,decision,*args,**kwargs)
    resolver.reassociate_contact=reassociate
    def initial_grounding(obs,decision,*args):
        return SimpleNamespace(decision=decision,object_mask=mask,part_mask=original_part,report={'kinds':['mask']})
    resolver.resolve_grasp=initial_grounding
    resolver.track=lambda *args,**kwargs:pytest.fail('unverified tracker must not enlarge occluded contact or payload surfaces')
    monkeypatch.setattr(runner,'Simulator',Sim)
    monkeypatch.setattr(runner,'RobotModel',Robot)
    monkeypatch.setattr(runner,'DepthMap',Map)
    monkeypatch.setattr(runner,'Planner',Plan)
    monkeypatch.setattr(runner,'VLM',Decisions)
    monkeypatch.setattr(runner.GroundingResolver,'from_config',lambda config:resolver)
    monkeypatch.setattr(runner,'Evaluator',lambda sim:SimpleNamespace(result=lambda:{
        'success':False,'collision':False,'first_grasp_success':False,'collision_free_success':False}))
    monkeypatch.setattr(runner,'grasp_candidates',lambda *args:([candidate],original_object.copy()))
    monkeypatch.setattr(runner,'part_cloud',lambda *args:part_reference.copy())
    monkeypatch.setattr(runner,'fit_table',lambda *args:.95)
    monkeypatch.setattr(runner,'nominal_grasp',lambda *args:None)
    def view_advice(*args,**kwargs):
        if 'required_points' in kwargs:
            assert len(kwargs['semantic_focus'])>=len(original_object)
        else:assert 'semantic_focus' not in kwargs
        return {'views':[[0.,55.,.8],[90.,55.,.8]] if multiple_mapping_views else [[0.,55.,.8]]}
    monkeypatch.setattr(runner,'suggest_views',view_advice)
    monkeypatch.setattr(runner,'save_observation',lambda *args:None)
    monkeypatch.setattr(runner,'time_parameterize',lambda path:path)
    monkeypatch.setattr(recording,'save_grounding_result',lambda *args:None)
    monkeypatch.setattr(runner,'save_episode',lambda out,sim,record,events:evidence.events.extend(events))
    if crop_mode=='fallback':
        def unavailable_crop(*args,**kwargs):raise ValueError('reference projection is outside current image')
        monkeypatch.setattr(runner,'crop_observation',unavailable_crop)
    result=runner.run_episode('occluded',3,'active',tmp_path,replay=tmp_path/'offline',
        config=runner.ExperimentConfig(max_observations=6 if multiple_mapping_views else 4,perception_backend='rgbd'))
    assert 'client_mutation' not in json.dumps(evidence.global_history)
    assert any('coverage_advice' in item for item in evidence.global_history)
    assert any(item.get('decision',{}).get('frame_id')==1 for item in evidence.global_history)

    if multiple_mapping_views:
        # Frames 2 and 3 belong to the SAME multi-view recovery loop. The old
        # end-of-loop call annotated only frame 3 and silently skipped frame 2;
        # frame 4 separately exercises the reserved residual-recovery branch.
        assert result['observations']==4 and result['vlm_calls']==7 and result['attempts']==0
        assert evidence.observations==[1,2,3,4] and len(evidence.cameras)==3
        assert evidence.association_frames==[2,3,4]
        assert evidence.planning_association_frames==[[],[2,3],[2,3,4]]
        assert evidence.targets[0].shape==(8,3) and all(len(points)==16 for points in evidence.targets[1:])
        assert all(np.array_equal(obj,original_object) and np.array_equal(part,part_reference)
            for obj,part in evidence.association_references)
        assert all(items is evidence.candidate_lists[0] for items in evidence.candidate_lists)
        assert np.array_equal(candidate.approach,original_ray)
        associated=[e for e in evidence.events if e['kind']=='contact_association']
        assert [e['frame_id'] for e in associated]==[2,3,4]
        assert all(e['preserved_grounding_frame']==1 and not e['grasp_ray_changed'] for e in associated)
        assert not evidence.executed and not evidence.removal_requests
        assert not (tmp_path/'occluded_active_0003/error.txt').exists()
        return

    assert result['observations']==(4 if post_close else 2)
    repairs=[e for e in evidence.events if e['kind']=='contact_annotation_repair']
    assert result['vlm_calls']==(6 if post_lift else 7 if post_close else 3)+len(repairs)
    assert result['attempts']==int(post_close)
    assert all(sum(r['frame_id']==frame for r in repairs)<=2 for frame in evidence.observations)
    assert evidence.observations==(list(range(1,5)) if post_close else [1,2])
    assert len(evidence.cameras)==(2 if post_lift else 3 if post_close else 1)
    assert evidence.candidate_lists[0] is evidence.candidate_lists[1]
    assert np.array_equal(candidate.approach,original_ray)
    assert len(evidence.targets[0])==8 # Explicit part was not broadened by the old 4 cm whole-object shortcut.
    assert evidence.vlm_calls[2][0]==2 and evidence.vlm_calls[2][1]['force_observe']
    context=evidence.vlm_calls[2][2][-1]['contact_reassociation']
    assert context['original_grounding_frame']==1 and context['frame_id']==2
    assert 'annotation only' in context['instructions']
    if crop_mode=='cropped':
        assert evidence.annotation_shapes==[(128,128)]*(1+len(repairs))
        assert evidence.association_shapes==[sample.depth.shape]*(1+len(repairs))
        assert len(evidence.targets[-1])==(8 if annotation_outcome in ('progress_then_reject','topology_then_reject','reference_then_reject') else 16)
    elif crop_mode=='fallback':
        assert evidence.annotation_shapes==evidence.association_shapes==[sample.depth.shape]
        assert any(e['kind']=='contact_annotation_crop_unavailable' for e in evidence.events)
    events=evidence.events
    replans=[e for e in events if e['kind']=='recovery_planning']
    assert len(replans)==1 and replans[0]['preserved_grounding_frame']==1
    if post_close:
        # The actual DepthMap removal operation keeps every 10 mm neighbor
        # endpoint, including when a bad annotation includes that neighbor.
        assert evidence.removal_requests
        assert all(np.min(cKDTree(neighbor).query(points)[0])>.009 for points in evidence.removal_requests)
        assert all(np.max(cKDTree(points).query(neighbor)[0])<1e-9 for points in evidence.remaining_obstacles)
        assert all(len(points)==8 for points in evidence.targets)
        if post_lift:
            assert 'lift' in evidence.executed and evidence.lift_checks==14
            measured=cKDTree(np.vstack(evidence.measured_objects))
            # Removal is supported by actually measured object points. The
            # camera's changed sampling differs from pure FK translation, so
            # deleting the predicted held cloud would fail this assertion.
            assert all(np.max(measured.query(points)[0])<1e-9 for points in evidence.removal_requests)
            if annotation_outcome=='accepted':
                assert result['failure']=='grounding: missing destination'
                assert not any(phase in evidence.executed for phase in ('lower','release','transfer'))
                assert np.min(evidence.targets[-1][:,2])>1.1
                assert any(e['kind']=='visual_verification' and e['lifted'] for e in events)
            else:
                assert result['failure']=='execution: visual lift verification failed'
                assert 'lower' in evidence.executed and 'release' in evidence.executed
                assert 'transfer' not in evidence.executed
                assert any(e['kind']=='contact_association_rejected' and e['frame_id']==4 for e in events)
            return
        assert 'close' in evidence.executed and 'lift' not in evidence.executed
        if annotation_outcome=='accepted':
            assert evidence.lift_checks==1 and result['failure']=='planning: lift deliberate lift rejection'
            assert len([e for e in events if e['kind']=='contact_association'])==3
        else:
            assert evidence.lift_checks==0
            assert result['failure']=='execution: post-close object/part association unverified; lift blocked'
            assert len([e for e in events if e['kind']=='contact_association_rejected'])==2+len(repairs)
        return
    if annotation_outcome in ('accepted','repair_then_accept','progress_then_accept','topology_then_accept','reference_then_accept'):
        assert len(evidence.targets)==2 and len(evidence.targets[-1])==16
        assert any(e['kind']=='contact_association' and not e['grasp_ray_changed'] for e in events)
        saved=json.loads((tmp_path/'occluded_active_0003/contact_association_002.json').read_text())
        assert saved['preserved_grounding_frame']==1 and saved['frame_id']==2
        assert saved['image_context']==context['image_context']
        if annotation_outcome=='repair_then_accept':
            assert len(repairs)==1 and len(evidence.annotation_shapes)==2
        elif annotation_outcome in ('progress_then_accept','topology_then_accept','reference_then_accept'):
            assert len(repairs)==2 and len(evidence.annotation_shapes)==3
    else:
        assert len(evidence.targets)==1
        assert any(e['kind']=='contact_association_rejected' for e in events)
        assert not any(e['kind']=='contact_association' for e in events)
        if annotation_outcome in ('progress_then_reject','topology_then_reject','reference_then_reject'):
            assert len(repairs)==2 and len(evidence.annotation_shapes)==3
    if topology_repair:
        failures=[e['diagnostics'] for e in events if e['kind']=='contact_association_rejected']
        assert failures[0]['anchor_pixels']>failures[1]['anchor_pixels']>=4
        assert [d['component_anchor_repairability']['anchor_reachable_fraction'] for d in failures[:2]]==[0.,1.]
        assert result['vlm_calls']==5 and result['observations']==2
    if annotation_outcome in ('reference_then_accept','reference_then_reject'):
        failures=[e['diagnostics'] for e in events if e['kind']=='contact_association_rejected']
        assert all(d['failure']=='reference_inconsistency' for d in failures)
        assert [d['anchor_pixels'] for d in failures[:2]]==[16,12]
        assert [d['rejection_counts']['reference_distance'] for d in failures[:2]]==[32,1]
        assert result['vlm_calls']==5 and result['observations']==2
    assert not (tmp_path/'occluded_active_0003/error.txt').exists()
