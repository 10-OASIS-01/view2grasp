"""An explicit fresh semantic transaction can reuse an already acquired image."""
from types import SimpleNamespace
import copy
import json
import numpy as np
import pytest

from view2grasp.geometry import orbit_pose
from view2grasp.types import AgentDecision,GraspCandidate,GroundingProposal,Observation


def ranking(**updates):
    result={'status':'complete','input_candidates':10,'checked_candidates':10,
        'incomplete_candidates':0,'nominal_clear_candidates':2,'nominal_blocked_candidates':8,
        'selected_priority_candidates':2,'priority_candidates_completed':2,
        'priority_pregrasp_ik_attempts':16,'priority_pregrasp_ik_solutions':0}
    result.update(updates)
    return result


@pytest.mark.parametrize('updates',[
    {'status':'time_budget'}, {'checked_candidates':9,'incomplete_candidates':1},
    {'selected_priority_candidates':1}, {'priority_candidates_completed':1},
    {'priority_pregrasp_ik_attempts':15}, {'priority_pregrasp_ik_solutions':1},
    {'nominal_clear_candidates':0,'nominal_blocked_candidates':0},
])
def test_incomplete_or_kinematically_valid_priority_set_cannot_trigger(updates):
    from view2grasp.runner import _current_frame_reselection_reason
    assert _current_frame_reselection_reason(ranking(**updates)) is None


def episode(tmp_path,monkeypatch,*,diagnostic=None,fresh_action='grasp',invalid=None,
            max_observations=9,ground_frame=3,refinement=False):
    from view2grasp import runner,recording
    e=SimpleNamespace(frames=[],groundings=[],candidates=[],calls=[],events=[],moves=[],advice=[],reachability=[])
    target=np.array([[.04+i*.001,-.12,.90] for i in range(20)])

    class Sim:
        def __init__(self,*args):
            self.q=np.zeros(7);self.gripper_q=np.array([.04,-.04]);self.frame=0
            self.view=(0.,80.,1.3);self.robot_xml='';self.joint_names=[];self.eef_name=''
        def observe(self):
            self.frame+=1;e.frames.append(self.frame)
            size=256 if refinement else 4
            return Observation(self.frame,np.full((size,size,3),self.frame,np.uint8),np.ones((size,size)),
                np.eye(3),orbit_pose(*self.view),self.q,self.gripper_q,0.,self.view)
        def move_camera(self,*view):self.view=tuple(view);e.moves.append(self.view)
        def validate_initial_proxies(self,*args):pass
        def step(self,*args):pytest.fail('failed planning never permits robot motion')
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
        def set_target(self,*args):pass
        def classify(self,p):return np.zeros(len(p),np.uint8)

    class Plan:
        def __init__(self,robot,map_,rng):
            self.rng=rng;self.checks=0;self.last_failure='unknown space'
            self.failure_counts={'pregrasp: unknown space':8}
            self.closest_invalid=None;self.closest_by_failure={};self.best_unknown=None
            self.geometry_ranking={}
        def choose(self,q,candidates,**kwargs):
            e.candidates.append(candidates);self.checks+=1
            # The same limitation remains after later mapping, exercising the
            # episode-wide one-shot cap without certifying any candidate.
            # Only the post-mapping retry has the complete geometric evidence;
            # a fresh candidate transaction starts without that evidence.
            self.geometry_ranking=(copy.deepcopy(diagnostic if diagnostic is not None else ranking())
                if len(e.candidates)>1 and candidates is e.candidates[-2] else {'status':'disabled'})
            return None
        def recovery_unknown(self):return target[:1]

    class Resolver:
        calls=0;seconds=0.;segmenter=None;sam_unavailable_reason='test'
        def resolve_grasp(self,obs,decision,*args):
            e.groundings.append((obs.frame_id,decision))
            assert decision.frame_id==obs.frame_id
            return SimpleNamespace(decision=decision,report={'kinds':['box']},object_mask=None,part_mask=None)
        def track(self,*args):return None

    def grasp(frame):
        box=(.25,.3,.65,.7) if refinement else (0.,0.,1.,1.)
        proposals=[GroundingProposal(kind='box',role=role,frame_id=frame,box=box)
            for role in ('object','grasp_part')]
        return AgentDecision(action='grasp',frame_id=frame,groundings=proposals,point=(.5,.5))

    class Decisions:
        calls=0;seconds=0.
        def __init__(self,*args,**kwargs):self.client=SimpleNamespace(close=lambda:None)
        def decide(self,obs,instruction,history,**kwargs):
            self.calls+=1;assert self.calls<=12
            e.calls.append({'frame_id':obs.frame_id,'kwargs':copy.deepcopy(kwargs),
                'history':json.loads(json.dumps(history)),'input_shape':obs.depth.shape})
            if 'grounding_refinement' in kwargs:
                assert kwargs['force_grasp'] is True
                if refinement=='error':raise RuntimeError('refinement API unavailable')
                if refinement=='abort':return AgentDecision(action='abort',frame_id=obs.frame_id,reason='uncertain crop')
                if refinement=='observe':return AgentDecision(action='observe',frame_id=obs.frame_id)
                fresh=AgentDecision(action='grasp',frame_id=obs.frame_id,point=(.75,.25),groundings=[
                    GroundingProposal(kind='box',role=role,frame_id=obs.frame_id,box=(.1,.1,.9,.9))
                    for role in ('object','grasp_part')])
                return fresh
            if 'current_frame_reselection' in kwargs:
                if fresh_action=='abort':return AgentDecision(action='abort',frame_id=obs.frame_id,reason='not visible')
                if fresh_action=='observe':
                    return AgentDecision(action='observe',frame_id=obs.frame_id,azimuth=-120,elevation=20,distance=.9)
                if invalid=='legacy':
                    return AgentDecision(action='grasp',frame_id=obs.frame_id,point=(.5,.5),bbox=(0.,0.,1.,1.),part_bbox=(0.,0.,1.,1.))
                if invalid=='stale':return grasp(obs.frame_id-1)
                if invalid=='camera':
                    return grasp(obs.frame_id).model_copy(update={'camera_id':'other'})
                if invalid=='source':
                    decision=grasp(obs.frame_id)
                    for proposal in decision.groundings:proposal.source='rgbd'
                    return decision
                return grasp(obs.frame_id)
            if kwargs.get('force_observe') or obs.frame_id<ground_frame:
                return AgentDecision(action='observe',frame_id=obs.frame_id,
                    azimuth=30*obs.frame_id,elevation=20,distance=.9)
            return grasp(obs.frame_id)

    def candidates(obs,*args):
        point=target[0]+[0.,0.,obs.frame_id*.001]
        ray=point-obs.T_wc[:3,3];ray/=np.linalg.norm(ray)
        x=np.cross([0.,0.,1.],ray);x/=np.linalg.norm(x)
        return [GraspCandidate(point,np.column_stack([x,np.cross(ray,x),ray]),.02,ray,1.,0.,
            surface_point=point,grasp_center=point)],target.copy()

    def advice(*args,budget,**kwargs):
        e.advice.append(budget)
        return {'views':[[30.,20.,.9]]*min(2,budget)}

    def reachable(robot,obs,contact,part,observed,**kwargs):
        assert np.allclose(contact,e.candidates[-1][0].surface_point)
        assert len(part) and observed is target
        assert kwargs['budget']==3 and kwargs['max_ik_calls']==72 and 0<=kwargs['time_budget']<=4.
        e.reachability.append(obs.frame_id)
        return {'views':[[120.,30.,.9]],'requires_new_grounding':True,
            'scope':'Kinematics only; no visibility or collision permission.'}

    for name,value in [('Simulator',Sim),('RobotModel',Robot),('DepthMap',Map),('Planner',Plan),('VLM',Decisions)]:
        monkeypatch.setattr(runner,name,value)
    monkeypatch.setattr(runner.GroundingResolver,'from_config',lambda config:Resolver())
    monkeypatch.setattr(runner,'Evaluator',lambda sim:SimpleNamespace(result=lambda:{
        'success':False,'collision':False,'first_grasp_success':False,'collision_free_success':False}))
    monkeypatch.setattr(runner,'nominal_grasp',lambda *args:None)
    monkeypatch.setattr(runner,'grasp_candidates',candidates)
    monkeypatch.setattr(runner,'part_cloud',lambda *args:target.copy())
    monkeypatch.setattr(runner,'fit_table',lambda *args:.8)
    monkeypatch.setattr(runner,'suggest_views',advice)
    monkeypatch.setattr(runner,'suggest_reachable_views',reachable)
    monkeypatch.setattr(runner,'check_vertical_part_midpoint',lambda obs,*args:{'frame_id':obs.frame_id,'status':'unresolved'})
    monkeypatch.setattr(runner,'save_observation',lambda *args:None)
    monkeypatch.setattr(recording,'save_grounding_result',lambda *args:None)
    monkeypatch.setattr(runner,'save_episode',lambda out,sim,record,events:e.events.extend(events))
    result=runner.run_episode('mug',3,'active',tmp_path,replay=tmp_path/'offline',
        config=runner.ExperimentConfig(max_observations=max_observations,refine_grasp_grounding=bool(refinement)))
    assert not (tmp_path/'mug_active_0003/error.txt').exists()
    return result,e


def test_mapping_image_can_start_one_explicit_fresh_grasp_without_extra_observation(tmp_path,monkeypatch):
    result,e=episode(tmp_path,monkeypatch)
    requests=[call for call in e.calls if 'current_frame_reselection' in call['kwargs']]
    assert len(requests)==1 and requests[0]['frame_id']==5
    assert [frame for frame,_ in e.groundings]==[3,5]
    assert e.frames==[1,2,3,4,5,6] and result['observations']==6 and result['attempts']==0
    assert e.candidates[0] is e.candidates[1]  # Old mapping preserves its ray.
    assert e.candidates[2] is e.candidates[3]  # New mapping preserves its NEW ray.
    assert e.candidates[0] is not e.candidates[2]
    assert not np.allclose(e.candidates[0][0].approach,e.candidates[2][0].approach)
    advice=requests[0]['kwargs']['current_frame_reselection']
    assert advice['retired_grounding_frame_id']==3 and advice['frame_id']==5
    assert advice['remaining_pregrasp_observations']==1 and advice['reserved_post_close_observations']==3
    assert advice['acquires_new_observation'] is False
    assert e.reachability==[5] and advice['kinematic_view_advice']['views']==[[120.,30.,.9]]
    assert advice['total_reselections']==1 and advice['current_frame_reselection_count']==1
    assert e.events[-1]['kind']=='observation_budget_reserved' and e.events[-1]['reserved_after_close']==3


@pytest.mark.parametrize('diagnostic',[
    {'status':'disabled'},ranking(status='time_budget'),
    ranking(priority_pregrasp_ik_solutions=1),ranking(priority_candidates_completed=1),
])
def test_unknown_or_incomplete_search_keeps_original_grounding_and_residual_mapping(tmp_path,monkeypatch,diagnostic):
    result,e=episode(tmp_path,monkeypatch,diagnostic=diagnostic)
    assert [frame for frame,_ in e.groundings]==[3]
    assert len(e.candidates)==3 and all(item is e.candidates[0] for item in e.candidates)
    assert not any('current_frame_reselection' in c['kwargs'] for c in e.calls)
    assert result['observations']==6 and result['attempts']==0


def test_complete_nominal_obstacles_are_only_a_reselection_reason_not_execution_permission(tmp_path,monkeypatch):
    result,e=episode(tmp_path,monkeypatch,diagnostic=ranking(nominal_clear_candidates=0,
        nominal_blocked_candidates=10,selected_priority_candidates=0))
    assert [frame for frame,_ in e.groundings]==[3,5] and result['attempts']==0
    assert any(row.get('reason')=='all nominal gripper approaches intersect observed obstacles' for row in e.events)


def test_no_mapping_allowance_means_no_current_frame_reselection(tmp_path,monkeypatch):
    result,e=episode(tmp_path,monkeypatch,max_observations=7)
    assert [frame for frame,_ in e.groundings]==[3] and result['observations']==4
    assert not any('current_frame_reselection' in c['kwargs'] for c in e.calls)
    assert e.events[-1]['reserved_after_close']==3


@pytest.mark.parametrize('invalid',['stale','legacy','camera','source'])
def test_reselection_cannot_reuse_old_or_nonsemantic_grounding(tmp_path,monkeypatch,invalid):
    result,e=episode(tmp_path,monkeypatch,invalid=invalid)
    assert [frame for frame,_ in e.groundings]==[3] and result['observations']==5
    assert e.events[-1]['kind']=='current_frame_reselection_rejected'
    assert result['attempts']==0


def test_reselection_abort_is_respected_without_acquiring_another_frame(tmp_path,monkeypatch):
    result,e=episode(tmp_path,monkeypatch,fresh_action='abort')
    assert result['failure']=='agent_abort: not visible' and result['observations']==5
    assert [frame for frame,_ in e.groundings]==[3]


def test_reselection_observe_uses_the_same_last_pregrasp_slot(tmp_path,monkeypatch):
    result,e=episode(tmp_path,monkeypatch,fresh_action='observe')
    assert result['observations']==6 and e.moves[-1]==(-120.,20.,.9)
    assert [frame for frame,_ in e.groundings]==[3,6]
    assert result['attempts']==0 and e.events[-1]['reserved_after_close']==3


def test_current_frame_reselection_is_once_even_with_more_remaining_mapping_allowance(tmp_path,monkeypatch):
    result,e=episode(tmp_path,monkeypatch,ground_frame=1)
    assert sum('current_frame_reselection' in c['kwargs'] for c in e.calls)==1
    assert [frame for frame,_ in e.groundings]==[1,3]
    assert result['observations']==6 and result['attempts']==0


def test_ten_observation_development_allowance_keeps_three_post_close_slots(tmp_path,monkeypatch):
    result,e=episode(tmp_path,monkeypatch,max_observations=10)
    assert result['observations']==7 and e.frames==list(range(1,8))
    assert e.events[-1]['kind']=='observation_budget_reserved' and e.events[-1]['reserved_after_close']==3
    assert sum('current_frame_reselection' in c['kwargs'] for c in e.calls)==1


def test_refinement_uses_same_frame_crop_and_restores_before_resolver_not_mapping(tmp_path,monkeypatch):
    result,e=episode(tmp_path,monkeypatch,refinement=True)
    calls=[call for call in e.calls if 'grounding_refinement' in call['kwargs']]
    assert [call['frame_id'] for call in calls]==[3,5]
    assert [frame for frame,_ in e.groundings]==[3,5]
    for call,(frame,decision) in zip(calls,e.groundings):
        metadata=call['kwargs']['grounding_refinement']
        assert metadata==call['kwargs']['derived_image_context']
        assert metadata['source_shape_hw']==[256,256] and metadata['new_observation'] is False
        assert list(call['input_shape'])==metadata['crop_shape_hw'] and max(call['input_shape'])<256
        x0,y0,x1,y1=metadata['bounds_xyxy']
        expected=(np.rint(np.array([.75,.25])*[x1-x0-1,y1-y0-1])+[x0,y0])/255
        assert np.allclose(decision.point,expected) and not np.allclose(decision.point,[.5,.5])
    assert e.candidates[0] is e.candidates[1] and e.candidates[2] is e.candidates[3]
    assert result['observations']==6 and result['attempts']==0
    assert sum(row['kind']=='grounding_refinement_decision' for row in e.events)==2


@pytest.mark.parametrize('reply',['error','observe','abort'])
def test_refinement_failure_never_falls_back_to_the_coarse_proposal(tmp_path,monkeypatch,reply):
    result,e=episode(tmp_path,monkeypatch,refinement=reply)
    assert not e.groundings and not e.candidates and result['observations']==3 and result['attempts']==0
    assert sum('grounding_refinement' in c['kwargs'] for c in e.calls)==1
    if reply=='abort':assert result['failure']=='agent_abort: uncertain crop'
    else:assert result['failure'].startswith('grounding refinement:')
