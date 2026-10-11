"""Offline integration smoke test. Mocked decisions never enter online metrics."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from view2grasp.types import AgentDecision,GraspCandidate,Observation


def test_observation_grounding_planner_recording_pipeline(tmp_path,monkeypatch):
    from view2grasp import runner
    class RecordedDecision:
        def __init__(self,*args,**kwargs):
            self.calls=0;self.seconds=0.;self.client=SimpleNamespace(close=lambda:None)
        def decide(self,obs,*args,**kwargs):
            self.calls+=1
            rgb=obs.rgb.astype(int)
            v,u=np.where((rgb[:,:,0]>100)&(rgb[:,:,0]>rgb[:,:,1]*2)&(rgb[:,:,0]>rgb[:,:,2]*2))
            h,w=obs.depth.shape
            bbox=(float(u.min()/(w-1)),float(v.min()/(h-1)),float(u.max()/(w-1)),float(v.max()/(h-1)))
            return AgentDecision(action='grasp',frame_id=obs.frame_id,target='test red pixels',part='test surface',
                point=(float(np.median(u)/(w-1)),float(np.median(v)/(h-1))),bbox=bbox,part_bbox=bbox)
    calls=[]
    def reject(self,q,candidates,**kwargs):
        calls.append(len(candidates));self.last_failure='deliberate test rejection';return None
    monkeypatch.setattr(runner,'VLM',RecordedDecision)
    monkeypatch.setattr(runner.Planner,'choose',reject)
    monkeypatch.setattr(runner,'save_episode',lambda *args:None)
    result=runner.run_episode('block',3,'fixed',tmp_path,replay=tmp_path/'offline',
        config=runner.ExperimentConfig(max_observations=2))
    assert calls and calls[0]>0
    assert result['online'] is False and not result['success'] and result['attempts']==0
    assert result['failure']=='planning: deliberate test rejection'
    assert (tmp_path/'block_fixed_0003/source_snapshot.zip').exists()


def _budget_episode(tmp_path,monkeypatch,*,recovery_views,planned_on=None,
                    strategy='active',max_observations=9,always_observe=False,task='mug'):
    """Exercise the runner's real recovery transaction without rendering or VLM IO."""
    from view2grasp import runner,recording
    evidence=SimpleNamespace(observed=[],executed=[],advice=[],events=[],planning_frames=[],vlm=[])
    points=np.array([[i*.001,0.,1.] for i in range(20)])
    grasp=GraspCandidate(np.zeros(3),np.eye(3),.02,np.array([0.,0.,1.]),1.,0.,
        surface_point=points[0],grasp_center=points[0])

    class Sim:
        def __init__(self,*args):
            self.q=np.zeros(7);self.gripper_q=np.array([.01,-.01]);self.phase='observe'
            self.robot_xml='';self.joint_names=[];self.eef_name='';self.frame=0
        def observe(self):
            self.frame+=1;evidence.observed.append((self.frame,self.phase))
            return Observation(self.frame,np.zeros((4,4,3),dtype=np.uint8),np.ones((4,4)),
                np.eye(3),np.eye(4),self.q.copy(),self.gripper_q.copy(),0.,(0.,55.,.8))
        def step(self,q,grip):
            self.q=np.asarray(q).copy();evidence.executed.append(self.phase)
        def move_camera(self,*args):pass
        def validate_initial_proxies(self,*args):pass
        def close(self):pass

    class Robot:
        def __init__(self,*args):pass
        def points(self,*args):return points[:1],None
        def volumes(self,*args):return ()
        def visual_volumes(self,*args):return ()
        def depth(self,obs):return np.ones_like(obs.depth)
        def pose(self,*args):return np.zeros(3),np.eye(3)
        def grasp_center_offset(self):return np.zeros(3)
        def ik(self,p,R,q):return q.copy()
        def set_q(self,*args):pass
        def close(self):pass

    class Map:
        def __init__(self,resolution):
            self.resolution=resolution;self.surface=points;self.views=[];self.table_z=0.
        def integrate(self,obs,*args):self.views.append(obs)
        def classify(self,p):return np.zeros(len(p),dtype=np.uint8)
        def certify_home(self,*args):pass
        def certify_robot_state(self,*args):pass
        def set_table(self,*args):pass
        def set_target(self,*args):pass
        def set_payload_source(self,*args):pass
        def remove_target_endpoints(self,*args):pass

    class Plan:
        def __init__(self,robot,map_,rng):
            self.rng=rng;self.calls=0;self.map=map_;self.checks=0
            self.failure_counts={};self.closest_invalid=None;self.closest_by_failure={}
            self.last_failure='unknown space';self.best_unknown=points[:1]
        def choose(self,q,candidates,**kwargs):
            self.calls+=1;self.checks=self.calls
            evidence.planning_frames.append(len(self.map.views))
            if planned_on is not None and self.calls>=planned_on:
                return candidates[0],np.array([q,q]),np.array([q,q])
            return None
        def recovery_unknown(self):return points[:1]
        def feasible_ik(self,p,R,q,**kwargs):
            # The mock lift remains blocked unless all reserved frames arrived.
            assert len(self.map.views)==max_observations
            return q.copy()
        def begin_payload_lift(self):pass
        def end_payload_lift(self):pass

    class Resolver:
        calls=0;seconds=0.;segmenter=None;sam_unavailable_reason='test'
        def resolve_grasp(self,obs,decision,*args):
            return SimpleNamespace(decision=decision,report={'kinds':['points']},
                object_mask=None,part_mask=None)
        def track(self,*args,**kwargs):return None

    class Decisions:
        def __init__(self,*args,**kwargs):
            self.calls=0;self.seconds=0.;self.client=SimpleNamespace(close=lambda:None)
        def decide(self,obs,*args,**kwargs):
            self.calls+=1;evidence.vlm.append((obs.frame_id,kwargs))
            if always_observe or kwargs.get('force_observe') or (obs.frame_id==1 and strategy=='active'):
                return AgentDecision(action='observe',frame_id=obs.frame_id)
            return AgentDecision(action='grasp',frame_id=obs.frame_id,point=(.5,.5),
                bbox=(0.,0.,1.,1.),part_bbox=(0.,0.,1.,1.))

    def advice(*args,budget,**kwargs):
        evidence.advice.append(budget)
        return {'views':[{} for _ in range(min(budget,recovery_views))]}

    monkeypatch.setattr(runner,'Simulator',Sim)
    monkeypatch.setattr(runner,'RobotModel',Robot)
    monkeypatch.setattr(runner,'DepthMap',Map)
    monkeypatch.setattr(runner,'Planner',Plan)
    monkeypatch.setattr(runner,'GroundingResolver',SimpleNamespace(from_config=lambda config:Resolver()))
    monkeypatch.setattr(runner,'VLM',Decisions)
    monkeypatch.setattr(runner,'Evaluator',lambda sim:SimpleNamespace(result=lambda:{
        'success':False,'collision':False,'first_grasp_success':False,'collision_free_success':False}))
    monkeypatch.setattr(runner,'grasp_candidates',lambda *args:([grasp],points.copy()))
    monkeypatch.setattr(runner,'part_cloud',lambda *args:points.copy())
    monkeypatch.setattr(runner,'cloud',lambda obs:points)
    monkeypatch.setattr(runner,'fit_table',lambda *args:0.)
    monkeypatch.setattr(runner,'suggest_views',advice)
    monkeypatch.setattr(runner,'nominal_grasp',lambda *args:None)
    monkeypatch.setattr(runner,'time_parameterize',lambda path:path)
    monkeypatch.setattr(runner,'save_observation',lambda *args:None)
    monkeypatch.setattr(recording,'save_grounding_result',lambda *args:None)
    monkeypatch.setattr(runner,'save_episode',lambda out,sim,record,events:evidence.events.extend(events))
    result=runner.run_episode(task,3,strategy,tmp_path,replay=tmp_path/'offline',
        config=runner.ExperimentConfig(max_observations=max_observations))
    return result,evidence


@pytest.mark.parametrize('recovery_views,planned_on,expected_planning_frames',[
    (4,2,[2,6]), # Full initial recovery allowance, then three post-close views.
    (3,3,[2,5,6]), # Residual follow-up consumes only the last pre-close frame.
])
def test_mug_mapping_recovery_preserves_three_post_close_views(
        tmp_path,monkeypatch,recovery_views,planned_on,expected_planning_frames):
    result,evidence=_budget_episode(tmp_path,monkeypatch,recovery_views=recovery_views,
        planned_on=planned_on)
    assert result['observations']==9 and result['attempts']==1 and not result['online']
    assert evidence.planning_frames==expected_planning_frames
    assert [frame for frame,phase in evidence.observed if phase=='close']==[7,8,9]
    assert 'lift' in evidence.executed
    post=[event for event in evidence.events if event['kind']=='postgrasp_residual']
    assert [event['postgrasp_view_index'] for event in post]==[1,2,3]
    retries=[event for event in evidence.events if event['kind'] in
        ('recovery_planning','retry_recovery_planning')]
    assert all(event['preserved_grounding_frame']==2 for event in retries)
    if recovery_views==3:
        assert evidence.advice[:2]==[4,1]
        episode=tmp_path/'mug_active_0003'
        previous=json.loads((episode/'planning_diagnostic_005_recovery.json').read_text())
        residual=json.loads((episode/'planning_diagnostic_006_residual.json').read_text())
        assert previous['checks']==2 and residual['checks']==3
        assert residual['recovery_unknown_points']==1
        assert residual.keys()==previous.keys()


@pytest.mark.parametrize('recovery_views',[3,4])
def test_mug_residual_unknown_stops_without_spending_post_close_reserve(
        tmp_path,monkeypatch,recovery_views):
    result,evidence=_budget_episode(tmp_path,monkeypatch,recovery_views=recovery_views)
    assert result['observations']==6 and result['attempts']==0
    assert result['failure']=='planning: unknown space'
    assert not evidence.executed
    assert evidence.events[-1]['kind']=='observation_budget_reserved'
    assert evidence.events[-1]['reserved_after_close']==3
    if recovery_views==3:
        residual=tmp_path/'mug_active_0003/planning_diagnostic_006_residual.json'
        assert json.loads(residual.read_text())['checks']==len(evidence.planning_frames)==3


def test_block_recovery_preserves_two_post_close_views(tmp_path,monkeypatch):
    result,evidence=_budget_episode(tmp_path,monkeypatch,recovery_views=1,planned_on=3,
        task='block',max_observations=6)
    assert result['observations']==6 and result['attempts']==1
    assert evidence.planning_frames==[2,3,4]
    assert [frame for frame,phase in evidence.observed if phase=='close']==[5,6]
    assert 'lift' in evidence.executed


@pytest.mark.parametrize('strategy',['fixed','random'])
def test_nonactive_recovery_uses_the_same_mug_budget(tmp_path,monkeypatch,strategy):
    result,evidence=_budget_episode(tmp_path,monkeypatch,recovery_views=4,strategy=strategy)
    assert result['observations']==6 and result['attempts']==0
    assert not evidence.executed
    assert evidence.events[-1]['reserved_after_close']==3


@pytest.mark.parametrize('max_observations,expected_preclose,expected_reserve',[(9,6,3),(2,1,1)])
def test_observe_decision_cannot_consume_reserved_frames(
        tmp_path,monkeypatch,max_observations,expected_preclose,expected_reserve):
    result,evidence=_budget_episode(tmp_path,monkeypatch,recovery_views=1,
        max_observations=max_observations,always_observe=True)
    assert result['observations']==expected_preclose and result['attempts']==0
    assert result['failure']=='observation budget reserved for post-grasp mapping'
    assert evidence.events[-1]['reserved_after_close']==expected_reserve
    assert evidence.vlm[-1][1]['force_grasp']
    assert not evidence.executed
