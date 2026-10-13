"""Pure IK recovery changes view; mapping recovery preserves its grasp ray."""
from types import SimpleNamespace
import json
import numpy as np
import pytest

from view2grasp.geometry import orbit_pose
from view2grasp.types import AgentDecision,GraspCandidate,Observation


def episode(tmp_path,monkeypatch,*,failure='ik',grounding_error=False,max_observations=5,repeated_view=False,
            reachable_views=True,semantic_error=False,ranking_override=None):
    from view2grasp import runner,recording
    evidence=SimpleNamespace(frames=[],groundings=[],candidates=[],calls=[],events=[],moves=[],reachability=[],semantics=[])
    target=np.array([[.04+i*.001,-.12,.90] for i in range(20)])

    class Sim:
        def __init__(self,*args):
            self.q=np.zeros(7);self.gripper_q=np.array([.04,-.04]);self.frame=0
            self.view=(0.,80.,1.3);self.robot_xml='';self.joint_names=[];self.eef_name=''
        def observe(self):
            self.frame+=1;evidence.frames.append(self.frame)
            return Observation(self.frame,np.full((4,4,3),self.frame,np.uint8),np.ones((4,4)),
                np.eye(3),orbit_pose(*self.view),self.q,self.gripper_q,0.,self.view)
        def move_camera(self,*view):self.view=tuple(view);evidence.moves.append(self.view)
        def validate_initial_proxies(self,*args):pass
        def step(self,*args):pytest.fail('no candidate in this test is certified for execution')
        def close(self):pass

    class Robot:
        def __init__(self,*args):pass
        def points(self,*args):return target[:1],None
        def volumes(self,*args):return ()
        def visual_volumes(self,*args):return ()
        def depth(self,obs):return np.full_like(obs.depth,10.)
        def pose(self,*args):return np.zeros(3),np.eye(3)
        def grasp_center_offset(self):return np.zeros(3)
        def close(self):pass

    class Map:
        def __init__(self,resolution):self.resolution=resolution;self.surface=target;self.table_z=.8
        def certify_home(self,*args):pass
        def integrate(self,*args):pass
        def set_table(self,*args):pass
        def set_target(self,points):evidence.target_part=points
        def classify(self,p):return np.zeros(len(p),np.uint8)

    class Plan:
        def __init__(self,robot,map_,rng):
            self.rng=rng;self.checks=19 # Earlier validity checks must not suppress pure IK recovery.
            self.failure_counts={'pregrasp IK':8};self.last_failure='pregrasp IK'
            self.closest_invalid=None;self.closest_by_failure={};self.best_unknown=None
        def choose(self,q,candidates,**kwargs):
            evidence.candidates.append(candidates)
            if failure=='geometry':
                self.checks+=1;self.last_failure='unknown space'
                self.failure_counts={'pregrasp: unknown space':8,'pregrasp IK':16}
                self.geometry_ranking={'status':'complete','input_candidates':20,
                    'checked_candidates':20,'incomplete_candidates':0,'nominal_clear_candidates':2,
                    'nominal_blocked_candidates':18,'selected_priority_candidates':2,
                    'priority_candidates_completed':2,'priority_pregrasp_ik_attempts':16,
                    'priority_pregrasp_ik_solutions':0}
                self.geometry_ranking.update(ranking_override or {})
            elif failure=='unknown':
                self.checks+=1;self.last_failure='unknown space'
                self.failure_counts={'pregrasp: unknown space':8}
            elif failure=='mixed':
                self.last_failure='pregrasp IK'
                self.failure_counts={'pregrasp IK':7,'pregrasp: observed obstacle':1}
            else:
                self.last_failure='planning budget exhausted: pregrasp IK' if failure=='budget' else 'pregrasp IK'
                self.failure_counts={'pregrasp IK':8}
            return None
        def recovery_unknown(self):return target[:1] if failure in ('unknown','geometry') else np.empty((0,3))

    class Resolver:
        calls=0;seconds=0.;segmenter=None;sam_unavailable_reason='test'
        def resolve_grasp(self,obs,decision,*args):
            evidence.groundings.append(obs.frame_id)
            if grounding_error and len(evidence.groundings)==1:
                raise ValueError('insufficient observed grasp-part geometry')
            return SimpleNamespace(decision=decision,report={'kinds':['box']},object_mask=None,part_mask=None)
        def track(self,*args):return None

    class Decisions:
        calls=0;seconds=0.
        def __init__(self,*args,**kwargs):self.client=SimpleNamespace(close=lambda:None)
        def decide(self,obs,instruction,history,**kwargs):
            self.calls+=1;assert self.calls<=6
            evidence.calls.append({'frame_id':obs.frame_id,'kwargs':kwargs,
                'history':json.loads(json.dumps(history))})
            if 'current_frame_reselection' in kwargs:
                return AgentDecision(action='abort',frame_id=obs.frame_id,reason='test stops second transaction')
            if kwargs.get('force_observe'):
                view=obs.view if repeated_view else (140.,15.,.9)
                return AgentDecision(action='observe',frame_id=obs.frame_id,
                    azimuth=view[0],elevation=view[1],distance=view[2])
            return AgentDecision(action='grasp',frame_id=obs.frame_id,point=(.5,.5),
                bbox=(0.,0.,1.,1.),part_bbox=(0.,0.,1.,1.))

    def candidates(obs,*args):
        # Re-grounding uses only the newly acquired image's camera-to-contact ray.
        point=target[0]+[0.,0.,obs.frame_id*.001]
        ray=point-obs.T_wc[:3,3];ray/=np.linalg.norm(ray)
        x=np.cross([0.,0.,1.],ray);x/=np.linalg.norm(x);y=np.cross(ray,x)
        return [GraspCandidate(point,np.column_stack([x,y,ray]),.02,ray,1.,0.,
            surface_point=point,grasp_center=point)],target.copy()

    def reachable(robot,obs,contact,part,observed,**kwargs):
        assert np.allclose(contact,evidence.candidates[-1][0].surface_point)
        assert part is evidence.target_part and np.isfinite(part).all()
        assert observed is target
        assert kwargs['budget']==3 and kwargs['max_ik_calls']==72
        assert 0<=kwargs['time_budget']<=4.
        evidence.reachability.append(obs.frame_id)
        # These are advisory candidates, not an instruction to move. The VLM's
        # independent camera decision below must still determine the new image.
        return {'views':[[120.,30.,.9]] if reachable_views else [],
            'source_frame_id':obs.frame_id,'requires_new_grounding':True,
            'scope':'Kinematics only; no scene clearance or grasp permission.'}

    def semantics(obs,*args,**kwargs):
        from view2grasp.grasp_semantics import GraspSemanticsError
        evidence.semantics.append(obs.frame_id)
        if semantic_error and (semantic_error=='always' or len(evidence.semantics)==1):
            raise GraspSemanticsError({'frame_id':obs.frame_id,'camera_id':obs.camera_id,
                'actual_view':list(obs.view),'status':'rejected','reason':'observed semantic contradiction'})
        return {'frame_id':obs.frame_id,'status':'unresolved'}

    monkeypatch.setattr(runner,'Simulator',Sim);monkeypatch.setattr(runner,'RobotModel',Robot)
    monkeypatch.setattr(runner,'DepthMap',Map);monkeypatch.setattr(runner,'Planner',Plan)
    monkeypatch.setattr(runner,'VLM',Decisions)
    monkeypatch.setattr(runner.GroundingResolver,'from_config',lambda config:Resolver())
    monkeypatch.setattr(runner,'Evaluator',lambda sim:SimpleNamespace(result=lambda:{
        'success':False,'collision':False,'first_grasp_success':False,'collision_free_success':False}))
    monkeypatch.setattr(runner,'nominal_grasp',lambda *args:None)
    monkeypatch.setattr(runner,'grasp_candidates',candidates)
    monkeypatch.setattr(runner,'part_cloud',lambda *args:target.copy())
    monkeypatch.setattr(runner,'fit_table',lambda *args:.8)
    monkeypatch.setattr(runner,'suggest_views',lambda *args,**kwargs:{'views':[[140.,15.,.9]]})
    monkeypatch.setattr(runner,'suggest_reachable_views',reachable)
    monkeypatch.setattr(runner,'check_vertical_part_midpoint',semantics)
    monkeypatch.setattr(runner,'save_observation',lambda *args:None)
    monkeypatch.setattr(recording,'save_grounding_result',lambda *args:None)
    monkeypatch.setattr(runner,'save_episode',lambda out,sim,record,events:evidence.events.extend(events))
    result=runner.run_episode('mug',3,'active',tmp_path,replay=tmp_path/'offline',
        config=runner.ExperimentConfig(max_observations=max_observations))
    assert not (tmp_path/'mug_active_0003/error.txt').exists()
    return result,evidence


def test_pure_ik_uses_fresh_view_then_new_grounding_despite_earlier_validity_checks(tmp_path,monkeypatch):
    result,e=episode(tmp_path,monkeypatch)
    assert e.frames==[1,2] and e.groundings==[1,2] and result['attempts']==0
    assert e.candidates[0] is not e.candidates[1]
    assert not np.allclose(e.candidates[0][0].approach,e.candidates[1][0].approach)
    request=e.calls[1]
    assert request['kwargs']=={'force_observe':True,'observe_purpose':'approach_reselection'}
    advice=request['history'][-1]['approach_reselection']
    assert advice['frame_id']==1 and advice['checks_delta']==0
    assert advice['failure_counts']=={'pregrasp IK':8}
    assert np.allclose(advice['observed_contact_world_m'],e.candidates[0][0].surface_point)
    assert np.allclose(advice['camera_to_contact_ray'],e.candidates[0][0].approach)
    assert advice['remaining_pregrasp_observations']==1 and advice['reserved_post_close_observations']==3
    assert advice['kinematic_view_advice']['views']==[[120.,30.,.9]]
    assert e.reachability==[1] and e.moves==[(140.,15.,.9)]
    assert all(row['checks']==19 and row['checks_delta']==0 for row in e.events if row['kind']=='planning')
    assert result['observations']==2 and e.events[-1]['kind']=='observation_budget_reserved'


def test_grounding_failure_repairs_same_frame_before_real_ik_reselection(tmp_path,monkeypatch):
    result,e=episode(tmp_path,monkeypatch,grounding_error=True)
    assert e.groundings==[1,1,2] and e.frames==[1,2]
    assert not e.calls[1]['kwargs'].get('force_observe')
    assert e.calls[2]['kwargs']['observe_purpose']=='approach_reselection'
    assert result['observations']==2


def test_semantic_part_contradiction_is_corrected_before_any_candidate_search(tmp_path,monkeypatch):
    result,e=episode(tmp_path,monkeypatch,semantic_error=True)
    assert e.groundings==[1,1,2] and e.semantics==[1,1,2]
    assert len(e.candidates)==2 and e.calls[1]['frame_id']==1
    assert not e.calls[1]['kwargs'].get('force_observe')
    assert any('grounding_rejection' in entry for entry in e.calls[1]['history'])
    assert result['observations']==2


def test_final_frame_semantic_correction_preserves_all_post_close_views(tmp_path,monkeypatch):
    result,e=episode(tmp_path,monkeypatch,max_observations=4,semantic_error=True)
    assert e.frames==[1] and e.groundings==[1,1] and len(e.candidates)==1
    assert e.calls[1]['kwargs']=={'force_grasp':True}
    assert not e.moves and result['observations']==1
    assert e.events[-1]['reserved_after_close']==3


def test_final_frame_semantic_repairs_stop_at_two_without_robot_motion(tmp_path,monkeypatch):
    result,e=episode(tmp_path,monkeypatch,max_observations=4,semantic_error='always')
    assert e.frames==[1] and e.groundings==[1,1,1] and len(e.calls)==3
    assert not e.candidates and not e.moves and result['attempts']==0


def test_grounding_rejection_does_not_reuse_stale_unknown_geometry(tmp_path,monkeypatch):
    result,e=episode(tmp_path,monkeypatch,failure='unknown',grounding_error=True)
    assert e.groundings[:2]==[1,1]
    assert not e.calls[1]['kwargs'].get('force_observe')
    assert e.calls[2]['kwargs']=={'force_observe':True}
    assert result['observations']==2


@pytest.mark.parametrize('failure',['mixed','budget'])
def test_mixed_or_incomplete_search_does_not_claim_pure_ik_failure(tmp_path,monkeypatch,failure):
    result,e=episode(tmp_path,monkeypatch,failure=failure)
    assert e.frames==[1] and not e.moves and result['attempts']==0
    assert not any(row['kind']=='approach_reselection_advice' for row in e.events)
    assert not any(call['kwargs'].get('force_observe') for call in e.calls)
    assert not e.reachability


def test_unknown_recovery_preserves_candidate_and_ray_within_same_budget(tmp_path,monkeypatch):
    result,e=episode(tmp_path,monkeypatch,failure='unknown')
    assert e.frames==[1,2] and e.groundings==[1]
    assert e.candidates[0] is e.candidates[1]
    assert e.calls[1]['kwargs']=={'force_observe':True}
    assert not any(row['kind']=='approach_reselection_advice' for row in e.events)
    assert not e.reachability
    assert result['observations']==2 and e.events[-1]['reserved_after_close']==3


def test_pure_ik_cannot_spend_post_close_reserve(tmp_path,monkeypatch):
    result,e=episode(tmp_path,monkeypatch,max_observations=4)
    assert result['observations']==1 and e.frames==[1] and len(e.calls)==1
    assert not e.moves and e.events[-1]['reserved_after_close']==3
    assert not e.reachability


def test_reselection_cannot_reuse_identical_view_as_new_ray_evidence(tmp_path,monkeypatch):
    result,e=episode(tmp_path,monkeypatch,repeated_view=True)
    assert e.frames==[1] and not e.moves and e.groundings==[1]
    assert result['failure']=='planning: pregrasp IK; approach recovery repeated the current view'


def test_empty_bounded_kinematic_advice_still_requires_new_image_and_grounding(tmp_path,monkeypatch):
    result,e=episode(tmp_path,monkeypatch,reachable_views=False)
    assert e.calls[1]['history'][-1]['approach_reselection']['kinematic_view_advice']['views']==[]
    assert e.reachability==[1] and e.frames==[1,2] and e.groundings==[1,2]
    assert result['attempts']==0 and result['observations']==2


@pytest.mark.parametrize('ranking_override,reason',[
    (None,'all nominally unblocked candidates exhausted pregrasp IK seeds'),
    ({'nominal_clear_candidates':0,'nominal_blocked_candidates':20,
        'selected_priority_candidates':0},'all nominal gripper approaches intersect observed obstacles'),
])
def test_complete_geometry_failure_requests_semantic_view_before_unknown_mapping(tmp_path,monkeypatch,ranking_override,reason):
    result,e=episode(tmp_path,monkeypatch,failure='geometry',ranking_override=ranking_override)
    assert e.calls[1]['kwargs']=={'force_observe':True,'observe_purpose':'approach_reselection'}
    advice=e.calls[1]['history'][-1]['approach_reselection']
    assert advice['reason']==reason and advice['checks_delta']==1
    assert advice['total_reselections']==1 and advice['approach_reselection_count']==1
    assert e.groundings==[1,2] and e.reachability==[1]
    assert not any(row['kind']=='recovery_view_advice' for row in e.events)
    assert result['observations']==2 and result['attempts']==0


@pytest.mark.parametrize('ranking_override',[
    {'status':'time_budget'}, {'incomplete_candidates':1,'checked_candidates':19},
    {'priority_candidates_completed':1}, {'priority_pregrasp_ik_solutions':1},
    {'selected_priority_candidates':1},
])
def test_partial_or_reachable_geometry_stays_mapping_only(tmp_path,monkeypatch,ranking_override):
    result,e=episode(tmp_path,monkeypatch,failure='geometry',ranking_override=ranking_override)
    assert e.calls[1]['kwargs']=={'force_observe':True}
    assert e.groundings==[1] and e.candidates[0] is e.candidates[1]
    assert not e.reachability and result['observations']==2


def test_two_reselection_kinds_are_each_bounded_once(tmp_path,monkeypatch):
    result,e=episode(tmp_path,monkeypatch,failure='geometry',max_observations=7)
    approach=[row for row in e.events if row['kind']=='approach_reselection_advice']
    current=[row for row in e.events if row['kind']=='current_frame_reselection_advice']
    assert len(approach)==len(current)==1
    assert approach[0]['total_reselections']==1 and current[0]['total_reselections']==2
    assert current[0]['approach_reselection_count']==current[0]['current_frame_reselection_count']==1
    assert e.reachability==[1,3] and result['observations']==3
    assert result['failure']=='agent_abort: test stops second transaction' and result['attempts']==0


def test_exhausted_pure_ik_reselection_cannot_fall_through_to_another_generic_view(tmp_path,monkeypatch):
    result,e=episode(tmp_path,monkeypatch,max_observations=7)
    assert e.frames==[1,2] and e.groundings==[1,2] and len(e.calls)==3
    assert e.reachability==[1] and len(e.moves)==1
    assert result['failure']=='planning: pregrasp IK; semantic view reselection limit reached'
    assert e.events[-1]['kind']=='approach_reselection_limit'
    assert e.events[-1]['total_reselections']==1 and result['attempts']==0
