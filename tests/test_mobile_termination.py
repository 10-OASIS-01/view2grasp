"""Terminal outcomes must not reuse an earlier, recoverable rejection."""
from dataclasses import asdict,replace
from types import SimpleNamespace
import json

import numpy as np
import pytest

from test_geometry import observation
from view2grasp.config import ExperimentConfig
from view2grasp.types import AgentDecision,GroundingProposal


@pytest.mark.parametrize('last_action,verified',[
    ('navigate',False),('navigate',True),('finish',False),('finish',True),
    ('abort',False),('rejected_navigation',False),
])
@pytest.mark.parametrize('initial_bad_point',[False,True])
@pytest.mark.parametrize('task',['NavigateKitchen','DeliverCanToBowl'])
def test_terminal_reason_after_rejected_grounding_and_successful_drive(
        monkeypatch,tmp_path,last_action,verified,initial_bad_point,task):
    """Run the actual orchestration and short-drive loop with offline devices.

    The first proposal is rejected, the second drives the base, and the last
    decision uses the final observation. Geometry policy and evaluator truth
    are mocked independently; no renderer, service, or old run is accessed.
    """
    import view2grasp.mobile_runner as runner

    simulations=[]
    class Sim:
        def __init__(self,*args,**kwargs):
            self.q=np.zeros(7);self.gripper_q=np.array([.02,-.02]);self.time=0.
            self.camera='robot0_agentview_left';self.base_pose=np.zeros(3)
            self.robot_xml='';self.joint_names=[];self.eef_name='';self.instruction='navigate'
            self.frame_id=0;self.phase='observe';self.verification_steps=0
            simulations.append(self)
        def auxiliary_state(self):return {}
        def camera_calibration(self,camera):
            # Calibration must not acquire a candidate image or change state.
            pose=np.eye(4);pose[:2,3]=self.base_pose[:2]
            return observation().K,pose
        def observe(self,camera=None):
            self.camera=camera or self.camera;self.frame_id+=1
            pose=np.eye(4);pose[:2,3]=self.base_pose[:2]
            obs=replace(observation(),frame_id=self.frame_id,camera_id=self.camera,
                T_wc=pose,q=self.q.copy(),gripper_q=self.gripper_q.copy(),timestamp=self.time,
                base_pose=self.base_pose.copy())
            if initial_bad_point:obs.depth[16,16]=1.2
            return obs
        def step(self,q,gripper,velocity=None):
            self.time+=.05;self.q=q.copy()
            if self.phase=='verify':self.verification_steps+=1
            if velocity is not None:
                c,s=np.cos(self.base_pose[2]),np.sin(self.base_pose[2])
                self.base_pose[:2]+=.05*np.array([[c,-s],[s,c]])@velocity[:2]
                self.base_pose[2]+=.05*velocity[2]
        def close(self):pass

    class Robot:
        def __init__(self,*args,**kwargs):pass
        def set_auxiliary_state(self,*args):pass
        def points(self,*args):return (np.array([[0.,0.,1.]]),)
        def boundary_points(self,*args):return np.array([[0.,0.,1.]])
        def volumes(self,*args):return []
        def visual_volumes(self,*args):return []
        def depth(self,obs):return np.full_like(obs.depth,10.)
        def close(self):pass

    class Map:
        surface=np.array([[x,y,1.] for x in np.linspace(-1,1,10) for y in np.linspace(-1,1,10)])
        def __init__(self,*args,**kwargs):pass
        def certify_home(self,*args):pass
        def integrate(self,*args):pass
        def certify_robot_state(self,*args):pass

    class Nav:
        def __init__(self,*args,**kwargs):pass
        def plan(self,start,end,**kwargs):return np.array([start,end])
        def edge(self,*args):return True

    decisions=[]
    class FakeVLM:
        calls=0;seconds=0.
        def __init__(self,*args):self.client=SimpleNamespace(close=lambda:None)
        def decide(self,obs,instruction,history):
            self.calls+=1
            assert self.calls<=3,'runner must stop at the terminal decision'
            if task=='NavigateKitchen':
                context=history[-1]['navigation_context']
                assert context['frame_id']==obs.frame_id and context['camera_id']==obs.camera_id
            elif self.calls==1:
                assert 'target_framing' not in history[-1]
            else:
                context=history[-1]['target_framing']
                assert context['source_frame_id']==1 and context['target_id']=='can'
                assert context['source_pixels']==[[23,16]]
                assert context['current_frame_id']==obs.frame_id
                assert context['source_age_frames']==obs.frame_id-1
                assert 'No candidate camera image was read' in context['scope']
            if self.calls==2 and initial_bad_point:
                rejected=next(item for item in reversed(history) if 'navigation_grounding' in item)
                assert rejected['navigation_grounding']['frame_id']==decisions[0].frame_id
                assert rejected['navigation_grounding']['pixel']==[16,16]
                assert rejected['navigation_grounding']['height_above_floor_m']==pytest.approx(.2)
                assert rejected['decision']==decisions[0].model_dump()
            action='navigate' if self.calls<3 else last_action
            groundings=[]
            if action=='navigate' and (self.calls>1 or initial_bad_point):
                groundings=[GroundingProposal(kind='points',role='navigation_goal',
                    frame_id=obs.frame_id,camera_id=obs.camera_id,
                    points=[(.5,.5) if self.calls==1 else (.9,.5)])]
            if action=='rejected_navigation':action='navigate'
            if task!='NavigateKitchen' and self.calls==1:
                groundings.append(GroundingProposal(kind='points',role='object',target_id='can',
                    frame_id=obs.frame_id,camera_id=obs.camera_id,points=[(.75,.5)]))
            decision=AgentDecision(action=action,frame_id=obs.frame_id,camera_id=obs.camera_id,
                groundings=groundings,reason='stop requested' if action=='abort' else '')
            decisions.append(decision)
            return decision

    def evaluator(sim):
        return SimpleNamespace(update=lambda:None,result=lambda:{
            'success':verified,'collision':False,'first_grasp_success':None,
            'official_success':verified,'collision_free_success':verified,
            'path_length_m':float(np.linalg.norm(sim.base_pose[:2])),'held_path_length_m':0.})
    resolver=SimpleNamespace(segmenter=None,sam_unavailable_reason='test',calls=0,seconds=0.)
    def save_result(out,sim,record,events):
        (out/'events.json').write_text(json.dumps(events))
        (out/'result.json').write_text(json.dumps(asdict(record)))

    monkeypatch.setattr(runner,'RoboCasaSimulator',Sim)
    monkeypatch.setattr(runner,'scene_eligibility',lambda sim:{'eligible':True})
    monkeypatch.setattr(runner,'MobileEvaluator',evaluator)
    monkeypatch.setattr(runner,'RobotModel',Robot)
    monkeypatch.setattr(runner,'DepthMap',Map)
    monkeypatch.setattr(runner,'Planner',lambda *args:SimpleNamespace())
    monkeypatch.setattr(runner,'NavigationCollision',lambda *args,**kwargs:None)
    monkeypatch.setattr(runner,'SE2Planner',Nav)
    monkeypatch.setattr(runner,'VLM',FakeVLM)
    monkeypatch.setattr(runner.GroundingResolver,'from_config',lambda config:resolver)
    monkeypatch.setattr(runner,'save_observation',lambda *args:None)
    monkeypatch.setattr(runner,'save_episode',save_result)

    result=runner.run_mobile_episode(task,3,'active',tmp_path,
        ExperimentConfig(navigation_observations=3,mobile_observations=3,perception_backend='rgbd'))
    out=tmp_path/f'{task}_active_0003'
    events=json.loads((out/'events.json').read_text())
    assert json.loads((out/'result.json').read_text())==result
    assert len(decisions)==3 and result['observations']==3
    assert result['path_length_m']>.2 and any(e['kind']=='navigation_plan' for e in events)
    rejection=('grounding: navigation goal is not on observed floor' if initial_bad_point else
        'grounding: navigation_goal required')
    assert result['geometry_rejections']==[rejection]+(
        ['grounding: navigation_goal required'] if last_action=='rejected_navigation' else [])
    if initial_bad_point:
        rejected=next(event for event in events if event['kind']=='rejected')
        assert rejected['navigation_grounding']['pixel']==[16,16]
    assert result['success'] is verified
    if verified:
        assert result['failure']==''
    elif last_action=='finish':
        assert result['failure']=='execution: independent success criterion not met after finish'
    elif last_action=='abort':
        assert result['failure']=='agent_abort: stop requested'
    else:
        assert result['failure']=='observation budget exhausted'
    terminal_events=[e for e in events if e['kind']=='observation_budget_exhausted']
    if last_action in ('navigate','rejected_navigation'):
        assert len(terminal_events)==1 and events[-1]==terminal_events[0]
        assert terminal_events[0]['observations']==terminal_events[0]['budget']==3
        if last_action=='navigate':
            assert events[-2]['kind']=='navigation_plan' and result['path_length_m']>.4
    else:
        assert terminal_events==[]
    assert simulations[0].verification_steps==(45 if last_action=='finish' else 0)
    assert not (out/'error.txt').exists()


@pytest.mark.parametrize('action',['navigate','turn'])
def test_countertop_only_blocks_both_navigation_grounding_and_turn_planning(monkeypatch,tmp_path,action):
    """Neither an image goal nor a direct turn may plan from countertop support."""
    import view2grasp.mobile_runner as runner

    events=[]
    class Sim:
        def __init__(self,*args,**kwargs):
            self.q=np.zeros(7);self.gripper_q=np.zeros(2);self.time=0.;self.base_pose=np.zeros(3)
            self.robot_xml='';self.joint_names=[];self.eef_name='';self.instruction='navigate'
            self.camera='robot0_nav_front';self.frame=0;self.phase='observe'
        def auxiliary_state(self):return {}
        def observe(self,camera=None):
            self.frame+=1;self.camera=camera or self.camera
            return replace(observation(),frame_id=self.frame,camera_id=self.camera)
        def step(self,*args):pytest.fail('unknown floor must block all physical motion')
        def close(self):pass

    class Robot:
        def __init__(self,*args,**kwargs):pass
        def set_auxiliary_state(self,*args):pass
        def points(self,*args):return (np.array([[0.,0.,.002]]),)
        def boundary_points(self,*args):return np.array([[0.,0.,.002],[0.,0.,1.2]])
        def volumes(self,*args):return []
        def visual_volumes(self,*args):return []
        def depth(self,obs):return np.full_like(obs.depth,10.)
        def close(self):pass

    class Map:
        surface=np.array([[x,y,.92] for x in np.linspace(-1,1,10) for y in np.linspace(-1,1,10)])
        def __init__(self,*args,**kwargs):pass
        def certify_home(self,*args):pass
        def integrate(self,*args):pass
        def certify_robot_state(self,*args):pass

    class Decisions:
        calls=0;seconds=0.
        def __init__(self,*args):self.client=SimpleNamespace(close=lambda:None)
        def decide(self,obs,instruction,history):
            self.calls+=1
            context=history[-1]['navigation_context']
            assert context['floor_samples']==[] and 'floor_z_m' not in context
            assert 'observe robot0_nav_floor' in context['floor_evidence_unavailable']
            if self.calls==2:
                assert any('navigation floor unobserved' in item.get('feedback','') for item in history)
                return AgentDecision(action='abort',frame_id=obs.frame_id,camera_id=obs.camera_id)
            assert self.calls==1
            return AgentDecision(action='navigate' if action=='navigate' else 'observe',
                frame_id=obs.frame_id,camera_id=obs.camera_id,
                observation_motion='turn_left' if action=='turn' else 'none',
                groundings=[GroundingProposal(kind='points',role='navigation_goal',
                    frame_id=obs.frame_id,camera_id=obs.camera_id,points=[(.5,.5)])])

    def no_planner(*args,**kwargs):pytest.fail('an unobserved floor must be rejected before constructing a navigation planner')
    evaluator=SimpleNamespace(update=lambda:None,result=lambda:{'success':False,'collision':False,
        'first_grasp_success':None,'official_success':False,'collision_free_success':False,
        'path_length_m':0.,'held_path_length_m':0.})
    monkeypatch.setattr(runner,'RoboCasaSimulator',Sim)
    monkeypatch.setattr(runner,'scene_eligibility',lambda sim:{'eligible':True})
    monkeypatch.setattr(runner,'MobileEvaluator',lambda sim:evaluator)
    monkeypatch.setattr(runner,'RobotModel',Robot);monkeypatch.setattr(runner,'DepthMap',Map)
    monkeypatch.setattr(runner,'Planner',lambda *args:SimpleNamespace())
    monkeypatch.setattr(runner,'NavigationCollision',no_planner);monkeypatch.setattr(runner,'SE2Planner',no_planner)
    monkeypatch.setattr(runner,'VLM',Decisions)
    monkeypatch.setattr(runner.GroundingResolver,'from_config',lambda config:
        SimpleNamespace(segmenter=None,sam_unavailable_reason='test',calls=0,seconds=0.))
    monkeypatch.setattr(runner,'save_observation',lambda *args:None)
    monkeypatch.setattr(runner,'save_episode',lambda out,sim,record,logged:events.extend(logged))
    result=runner.run_mobile_episode('NavigateKitchen',3,'active',tmp_path,
        ExperimentConfig(navigation_observations=2,perception_backend='rgbd'))
    assert result['observations']==2 and result['path_length_m']==0.
    assert len(result['geometry_rejections'])==1 and 'navigation floor unobserved' in result['geometry_rejections'][0]
    assert not any(event['kind']=='navigation_plan' for event in events)
    assert not (tmp_path/'NavigateKitchen_active_0003/error.txt').exists()
