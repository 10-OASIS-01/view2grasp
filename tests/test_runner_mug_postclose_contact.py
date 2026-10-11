"""Only strict current post-close semantics can add mug finger contact."""
import copy
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree

from test_contact_reassociation import scene,annotations
from test_object_ownership import annotation as ownership_annotation
from view2grasp.grounding import GroundingResolver,ContactAssociationError
from view2grasp.mapping import OCCUPIED
from view2grasp.object_ownership import resolve_object_ownership
from view2grasp.payload_registration import PayloadRegistrationError
from view2grasp.types import AgentDecision,GraspCandidate


def contains(points,other):
    return len(points)>0 and np.all(cKDTree(points).query(other)[0]<1e-10)


def run_case(tmp_path,monkeypatch,*,mode='first_success',ownership=True,preclose_recovery=False,
             owned=True):
    from view2grasp import runner,recording
    obs,xyz,object_mask,original=scene()
    initial_part=np.zeros_like(object_mask);initial_part[10:14,8:10]=True
    current_part=np.zeros_like(object_mask);current_part[10:14,8:11]=True
    measured=xyz[object_mask];allowed_part=xyz[current_part]
    extra=xyz[11,11][None]
    neighbor=xyz[10:14,12:16].reshape(-1,3)
    position=np.array([.2,-.3,.8])
    rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    grasp=GraspCandidate(np.zeros(3),np.eye(3),.02,np.array([0.,0.,1.]),0.,0.,
        surface_point=original[0],grasp_center=original[0])
    e=SimpleNamespace(events=[],requests=[],association_calls=[],targets=[],advice=[],
        retired=[],removals=[],certificates=[],observations=[],physical_steps=0,
        ownership_frames=[],original=original,measured=measured,allowed_part=allowed_part,
        extra=extra,position=position,rotation=rotation)

    class Sim:
        def __init__(self,*args):
            self.q=np.zeros(7);self.gripper_q=np.array([.02,-.02]);self.phase='observe'
            self.robot_xml='';self.joint_names=[];self.eef_name='';self.frame=0;e.sim=self
        def observe(self):
            self.frame+=1;e.observations.append((self.frame,self.phase,e.physical_steps))
            return replace(obs,frame_id=self.frame,q=self.q.copy(),gripper_q=self.gripper_q.copy())
        def move_camera(self,*args):pass
        def validate_initial_proxies(self,*args):pass
        def step(self,q,grip):self.q=np.array(q);e.physical_steps+=1
        def close(self):pass

    class Robot:
        def __init__(self,*args):pass
        def points(self,*args):return np.zeros((1,3)),np.zeros(1,bool)
        def volumes(self,*args):return ()
        def visual_volumes(self,*args):return ()
        def depth(self,sample):return np.full_like(sample.depth,10.)
        def pose(self,*args):return position.copy(),rotation.copy()
        def grasp_center_offset(self):return np.zeros(3)
        def ik(self,p,R,q):return q.copy()
        def set_q(self,*args):pass
        def close(self):pass

    class Map(runner.DepthMap):
        def __init__(self,resolution):
            super().__init__(resolution,workspace_bounds=None);e.map=self
        def set_table(self,z):self.table_z=z
        def certify_home(self,*args):pass
        def certify_robot_state(self,*args):pass
        def integrate(self,sample,*args):
            self.views.append(sample)
            observed=measured if e.sim.phase=='close' else original
            self.surface=np.unique(np.vstack([self.surface,observed,neighbor]),axis=0)
            self.tree=cKDTree(self.surface);self._refresh_non_target()
        def classify(self,points):return np.zeros(len(points),np.uint8)
        def set_target(self,points):
            e.targets.append((e.sim.phase,points.copy()));super().set_target(points)
        def set_payload_source(self,points):
            e.certificates.append(points.copy());super().set_payload_source(points)
        def remove_target_endpoints(self,points,radius=None):
            e.removals.append(points.copy());super().remove_target_endpoints(points,radius)
        def retire_owned_source(self,owned,sample):
            world=e.planner.payload_collision_local()@rotation.T+position
            assert contains(world,owned.object_points)
            assert np.array_equal(owned.object_points,original)
            e.retired.append(sample.frame_id)
            return super().retire_owned_source(owned,sample)

    class Plan:
        add_observed_payload=runner.Planner.add_observed_payload
        payload_collision_local=runner.Planner.payload_collision_local
        def __init__(self,robot,map_,rng):
            self.rng=rng;self.checks=0;self.last_failure='';self.failure_counts={}
            self.closest_invalid=None;self.closest_by_failure={};self.payload_invalid=None
            self.attached=None;self.observed_payload_local=None;self.registered_payload_local=None
            self.choose_calls=0;self.best_unknown=None;e.planner=self
        def choose(self,q,candidates,**kw):
            self.choose_calls+=1;self.checks+=1
            if preclose_recovery and self.choose_calls==1:
                self.last_failure='unknown space';self.failure_counts={'unknown space':1};return None
            return grasp,np.array([q,q]),np.array([q,q])
        def recovery_unknown(self):return original[:1] if preclose_recovery else np.empty((0,3))
        def begin_payload_lift(self):pass
        def end_payload_lift(self):pass
        def feasible_ik(self,*args,**kwargs):self.last_failure='strict test lift stop';return None

    class Decisions:
        def __init__(self,*args,**kwargs):
            self.calls=0;self.seconds=0.;self.client=SimpleNamespace(close=lambda:None)
            self.counts=Counter()
        def decide(self,sample,instruction,history,**options):
            self.calls+=1
            if options.get('observe_purpose')=='annotation':
                self.counts[sample.frame_id]+=1
                e.requests.append({'frame':sample.frame_id,'phase':e.sim.phase,'steps':e.physical_steps,
                    'image':sample,'instruction':instruction,'history':copy.deepcopy(history),'options':options})
                first_close=next(frame for frame,phase,_ in e.observations if phase=='close')
                if mode=='first_success' and sample.frame_id!=first_close:
                    return AgentDecision(action='observe',frame_id=sample.frame_id)
                proposal_mask=object_mask.copy()
                if mode=='reject':proposal_mask[10:14,12:16]=True
                result=annotations(sample,proposal_mask,current_part,target_id='mug')
                return result.model_copy(update={'action':mode}) if mode in ('abort','grasp') else result
            if options.get('force_observe'):
                return AgentDecision(action='observe',frame_id=sample.frame_id)
            return annotations(sample,initial_part,target_id='mug').model_copy(update={
                'action':'grasp','target':'mug','part':'outer vertical handle bar'})

    real=GroundingResolver();association_counts=Counter()
    def resolve_grasp(sample,decision,*args):
        return SimpleNamespace(decision=decision,object_mask=initial_part,part_mask=initial_part,
            report={'kinds':['mask']})
    def associate(sample,decision,object_reference,part_reference,target_id,support,robot_depth,**options):
        assert np.array_equal(object_reference,original) and np.array_equal(part_reference,original)
        assert target_id=='mug' and support==.95 and options=={'source_supported':False}
        e.association_calls.append((sample.frame_id,e.sim.phase,e.physical_steps))
        association_counts[sample.frame_id]+=1
        if mode=='progress_reject':
            count=4-association_counts[sample.frame_id]
            raise ContactAssociationError('strict synthetic rejection',{
                'frame_id':sample.frame_id,'camera_id':sample.camera_id,'role':'object',
                'anchor_pixels':4,'rejected_pixel_count':count,'failure':'reference_inconsistency',
                'coordinate_space':'source_image_normalized_pixel_centers',
                'source_shape_hw':list(sample.depth.shape),'rejected_samples':[],
                'reference_projection':{'bbox_normalized':None}})
        return real.reassociate_contact(sample,decision,object_reference,part_reference,target_id,
            support,robot_depth,**options)
    resolver=SimpleNamespace(calls=0,seconds=0.,segmenter=None,sam_unavailable_reason='test',resolve_grasp=resolve_grasp,
        track=lambda *args,**kwargs:original.copy(),reassociate_contact=associate)
    def annotate(vlm,sample,reference,identity,support,**options):
        e.ownership_frames.append(sample.frame_id)
        assert contains(original,reference) and contains(reference,original)
        result=(resolve_object_ownership(sample,ownership_annotation(sample,initial_part,seeds=((8,10),)),
            reference,identity,support,options['robot_depth'],allow_partial_seed_components=True) if owned else None)
        return SimpleNamespace(ownership=result,report={'semantic_requests':1,'vlm_calls':0})
    def register(*args):raise PayloadRegistrationError('test registration abstention',{'accepted':False})
    def advice(*args,**kwargs):
        e.advice.append({'phase':e.sim.phase,'coverage':np.array(args[2]),
            'occluders':kwargs.get('observed_points',np.empty((0,3))).copy()})
        return {'views':[[0.,55.,.8]]}
    def no_crop(*args,**kwargs):raise ValueError('test uses the original acquired image')
    for name,value in [('Simulator',Sim),('RobotModel',Robot),('DepthMap',Map),('Planner',Plan),
            ('VLM',Decisions),('annotate_source_object',annotate),('register_payload_source',register),
            ('suggest_views',advice),('crop_observation',no_crop)]:monkeypatch.setattr(runner,name,value)
    monkeypatch.setattr(runner.GroundingResolver,'from_config',lambda config:resolver)
    monkeypatch.setattr(runner,'Evaluator',lambda sim:SimpleNamespace(result=lambda:{
        'success':False,'collision':False,'first_grasp_success':False,'collision_free_success':False}))
    monkeypatch.setattr(runner,'grasp_candidates',lambda *args:([grasp],original.copy()))
    monkeypatch.setattr(runner,'part_cloud',lambda *args:original.copy())
    monkeypatch.setattr(runner,'fit_table',lambda *args:.95)
    monkeypatch.setattr(runner,'check_vertical_part_midpoint',lambda *args,**kw:{'status':'test'})
    monkeypatch.setattr(runner,'nominal_grasp',lambda *args:None)
    monkeypatch.setattr(runner,'time_parameterize',lambda path:path)
    monkeypatch.setattr(runner,'save_observation',lambda *args:None)
    monkeypatch.setattr(recording,'save_grounding_result',lambda *args:None)
    monkeypatch.setattr(runner,'save_episode',lambda out,sim,record,events:e.events.extend(events))
    maximum=5 if preclose_recovery else 4
    result=runner.run_episode('mug',3,'active',tmp_path,replay=tmp_path/'offline',
        config=runner.ExperimentConfig(max_observations=maximum,source_object_ownership=ownership))
    assert result['failure']=='planning: lift strict test lift stop'
    assert result['observations']==maximum and result['attempts']==1
    assert not result['success'] and not result['online']
    e.result=result;e.output=tmp_path/'mug_active_0003'
    return e


def test_verified_part_and_measured_object_have_separate_postclose_lifecycles(tmp_path,monkeypatch):
    e=run_case(tmp_path,monkeypatch)
    # A later failed/omitted same-static-epoch annotation cannot revoke the
    # first success or substitute a whole tracked object as a permitted part.
    assert [r['frame'] for r in e.requests]==[2,3,4]
    assert len(e.association_calls)==1
    for request in e.requests:
        assert request['phase']=='close' and request['steps']==e.physical_steps
        assert len(request['history'])==1
        assert request['options']['force_observe'] is True
        assert 'far outer vertical handle segment, away from the cup body' in request['instruction']
        assert 'Exclude both horizontal handle connectors and both junctions' in request['instruction']
        assert 'polygons may be IDENTICAL' in request['instruction']
        assert 'SAME clearly visible patch of that permitted outer bar' in request['instruction']
        assert 'Neither polygon needs to cover the full handle or cup' in request['instruction']
        assert 'Omit regions if the permitted outer bar is occluded or uncertain' in request['instruction']
        assert 'earlier overhead view' in request['instruction']
    assert contains(e.targets[-1][1],e.allowed_part)
    assert contains(e.allowed_part,e.targets[-1][1])
    assert not contains(e.targets[-1][1],e.extra)
    expected=(e.allowed_part-e.position)@e.rotation
    assert contains(e.planner.attached_contact_local,expected)
    assert contains(expected,e.planner.attached_contact_local)
    world=e.planner.payload_collision_local()@e.rotation.T+e.position
    assert contains(world,e.measured) and contains(e.certificates[-1],e.measured)
    assert not contains(e.planner.attached@e.rotation.T+e.position,e.extra)
    assert contains(e.advice[2]['occluders'],e.extra)
    assert contains(e.advice[2]['coverage'],e.extra+[0.,0.,.13])
    # The extra measured object face was neither made an ownership input nor
    # added to the legacy broad retirement target. Its endpoint stays occupied
    # even though it is now represented as actual payload source evidence.
    assert e.ownership_frames==[1,2,3,4] and e.retired==[2,3,4]
    assert len(e.removals)==2 and all(not contains(p,e.extra) for p in e.removals)
    assert contains(e.map.surface,e.extra)
    from view2grasp.mapping import DepthMap
    assert DepthMap.classify(e.map,e.extra)[0]==OCCUPIED
    assert len([x for x in e.events if x['kind']=='postclose_contact_evidence'])==1


@pytest.mark.parametrize('mode,answers',[('reject',2),('progress_reject',3)])
def test_failed_same_frame_corrections_never_add_contact_or_collision_evidence(tmp_path,monkeypatch,mode,answers):
    e=run_case(tmp_path,monkeypatch,mode=mode)
    assert Counter(r['frame'] for r in e.requests)=={2:answers,3:answers,4:answers}
    assert all(np.array_equal(points,e.original) for _,points in e.targets)
    assert not contains(e.planner.payload_collision_local()@e.rotation.T+e.position,e.extra)
    assert not contains(e.certificates[-1],e.extra)
    assert e.retired==[2,3,4] and len(e.removals)==2
    assert not any(x['kind']=='postclose_contact_evidence' for x in e.events)


def test_postclose_permission_does_not_activate_mug_preclose_annotation(tmp_path,monkeypatch):
    e=run_case(tmp_path,monkeypatch,preclose_recovery=True)
    assert e.planner.choose_calls==2
    assert [phase for _,phase,_ in e.observations]==['observe','observe','close','close','close']
    assert [r['frame'] for r in e.requests]==[3,4,5]
    assert all(frame>=3 and phase=='close' for frame,phase,_ in e.association_calls)
    assert e.ownership_frames==[1,3,4,5]


def test_default_nonownership_mug_never_requests_new_semantic_contact(tmp_path,monkeypatch):
    e=run_case(tmp_path,monkeypatch,ownership=False)
    assert not e.requests and not e.association_calls and not e.ownership_frames
    assert not any(x['kind']=='postclose_contact_evidence' for x in e.events)


@pytest.mark.parametrize('action',['abort','grasp'])
def test_nonannotation_action_cannot_grant_contact_even_with_valid_regions(tmp_path,monkeypatch,action):
    e=run_case(tmp_path,monkeypatch,mode=action)
    assert [r['frame'] for r in e.requests]==[2,3,4]
    assert not e.association_calls
    assert all(np.array_equal(points,e.original) for _,points in e.targets)
    assert not contains(e.certificates[-1],e.extra)


def test_contact_success_cannot_replace_abstained_ownership_or_retire_its_endpoints(tmp_path,monkeypatch):
    e=run_case(tmp_path,monkeypatch,owned=False)
    assert contains(e.targets[-1][1],e.allowed_part)
    assert contains(e.planner.payload_collision_local()@e.rotation.T+e.position,e.measured)
    assert contains(e.certificates[-1],e.measured)
    assert not e.retired and len(e.removals)==2
    assert contains(e.map.surface,e.extra)
    from view2grasp.mapping import DepthMap
    assert DepthMap.classify(e.map,e.extra)[0]==OCCUPIED
