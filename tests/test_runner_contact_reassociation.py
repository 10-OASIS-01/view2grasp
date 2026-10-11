"""Exercise bounded annotation-only recovery through the real runner loop."""
from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest

from test_contact_reassociation import annotations,scene
from view2grasp.grounding import GroundingResolver
from view2grasp.types import GraspCandidate


@pytest.mark.parametrize('annotation_outcome',['accepted','missing_part','api_failure'])
def test_mapping_reassociation_preserves_ray_and_never_moves_for_annotation(
        tmp_path,monkeypatch,annotation_outcome):
    from view2grasp import runner,recording
    sample,xyz,mask,part_reference=scene()
    original_part=np.zeros_like(mask);original_part[10:14,8:10]=True
    original_object=xyz[mask]
    evidence=SimpleNamespace(cameras=[],observations=[],targets=[],candidate_lists=[],vlm_calls=[],events=[])
    candidate=GraspCandidate(np.zeros(3),np.eye(3),.02,np.array([0.,0.,1.]),1.,0.,
        surface_point=part_reference[0],grasp_center=part_reference[0])
    original_ray=candidate.approach.copy()

    class Sim:
        def __init__(self,*args):
            self.q=np.zeros(7);self.gripper_q=np.array([.04,-.04]);self.frame=0
            self.robot_xml='';self.joint_names=[];self.eef_name=''
        def observe(self):
            self.frame+=1;evidence.observations.append(self.frame)
            return replace(sample,frame_id=self.frame)
        def move_camera(self,*view):evidence.cameras.append(view)
        def validate_initial_proxies(self,*args):pass
        def step(self,*args):pytest.fail('this episode must not execute an uncertified grasp')
        def close(self):pass

    class Robot:
        def __init__(self,*args):pass
        def points(self,*args):return np.zeros((1,3)),None
        def volumes(self,*args):return ()
        def visual_volumes(self,*args):return ()
        def depth(self,obs):return np.full_like(obs.depth,10.)
        def pose(self,*args):return np.zeros(3),np.eye(3)
        def grasp_center_offset(self):return np.zeros(3)
        def close(self):pass

    class Map:
        def __init__(self,resolution):
            self.resolution=resolution;self.surface=np.vstack([original_object,xyz[10:14,12:16].reshape(-1,3)])
            self.table_z=.95;self.views=[]
        def certify_home(self,*args):pass
        def integrate(self,obs,*args):self.views.append(obs)
        def set_table(self,z):self.table_z=z
        def classify(self,points):return np.zeros(len(points),np.uint8)
        def set_target(self,points):evidence.targets.append(points.copy())

    class Plan:
        def __init__(self,robot,map_,rng):
            self.rng=rng;self.calls=0;self.checks=0;self.last_failure='unknown space'
            self.failure_counts={};self.closest_invalid=None;self.closest_by_failure={}
            self.best_unknown=original_object[:1]
        def recovery_unknown(self):return original_object[:1]
        def choose(self,q,candidates,**kwargs):
            evidence.candidate_lists.append(candidates)
            self.calls+=1;self.checks+=1
            return None

    class Decisions:
        def __init__(self,*args,**kwargs):
            self.calls=0;self.seconds=0.;self.client=SimpleNamespace(close=lambda:None)
        def decide(self,obs,instruction,history,**kwargs):
            self.calls+=1;evidence.vlm_calls.append((obs.frame_id,kwargs,json.loads(json.dumps(history))))
            assert self.calls<=3,'one annotation call must not create an unbounded repair loop'
            if self.calls==1:
                return annotations(obs,mask,original_part).model_copy(update={'action':'grasp'})
            if self.calls==2:
                return annotations(obs,mask).model_copy(update={'groundings':[]})
            if annotation_outcome=='api_failure':raise RuntimeError('VLM service unavailable in mock')
            result=annotations(obs,mask)
            if annotation_outcome=='missing_part':result=result.model_copy(update={'groundings':result.groundings[:1]})
            return result

    resolver=GroundingResolver()
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
    monkeypatch.setattr(runner,'suggest_views',lambda *args,**kwargs:{'views':[[0.,55.,.8]]})
    monkeypatch.setattr(runner,'save_observation',lambda *args:None)
    monkeypatch.setattr(recording,'save_grounding_result',lambda *args:None)
    monkeypatch.setattr(runner,'save_episode',lambda out,sim,record,events:evidence.events.extend(events))
    result=runner.run_episode('occluded',3,'active',tmp_path,replay=tmp_path/'offline',
        config=runner.ExperimentConfig(max_observations=4,perception_backend='rgbd'))

    assert result['observations']==2 and result['vlm_calls']==3 and result['attempts']==0
    assert evidence.observations==[1,2] and len(evidence.cameras)==1
    assert evidence.candidate_lists[0] is evidence.candidate_lists[1]
    assert np.array_equal(candidate.approach,original_ray)
    assert len(evidence.targets[0])==8 # Explicit part was not broadened by the old 4 cm whole-object shortcut.
    assert evidence.vlm_calls[2][0]==2 and evidence.vlm_calls[2][1]['force_observe']
    context=evidence.vlm_calls[2][2][-1]['contact_reassociation']
    assert context['original_grounding_frame']==1 and context['frame_id']==2
    assert 'annotation only' in context['instructions']
    events=evidence.events
    replans=[e for e in events if e['kind']=='recovery_planning']
    assert len(replans)==1 and replans[0]['preserved_grounding_frame']==1
    if annotation_outcome=='accepted':
        assert len(evidence.targets)==2 and len(evidence.targets[-1])==16
        assert any(e['kind']=='contact_association' and not e['grasp_ray_changed'] for e in events)
        saved=json.loads((tmp_path/'occluded_active_0003/contact_association_002.json').read_text())
        assert saved['preserved_grounding_frame']==1 and saved['frame_id']==2
    else:
        assert len(evidence.targets)==1
        assert any(e['kind']=='contact_association_rejected' for e in events)
        assert not any(e['kind']=='contact_association' for e in events)
    assert not (tmp_path/'occluded_active_0003/error.txt').exists()
