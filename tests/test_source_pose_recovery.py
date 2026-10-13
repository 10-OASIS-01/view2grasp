"""An unused real view may resolve source pose; never waive observed obstacles."""
from dataclasses import replace
import copy
import json
import numpy as np
import pytest
from test_geometry import observation
from test_source_ownership_lifecycle import run_lifecycle,_contains
from test_vlm import isolated_api_settings,retry_sleeps
from test_vlm_object_annotation import client_with_responses
from view2grasp.postclose_preview import source_pose_recovery
from view2grasp.types import AgentDecision


def inputs():
    ref=replace(observation(),frame_id=2,view=(-165.,15.,.8))
    current=replace(ref,frame_id=9,view=(-90.,10.,1.3))
    preview={'mapping_evidence_available':False,'occupied_unresolved':True}
    return dict(preview=preview,failure='ambiguous_local_pose',reference=ref,current=current,
        acquired_views=[(60.,10.,1.3),current.view],remaining=1,used=False)


def test_one_explicit_source_revisit_uses_only_real_reference_and_existing_budget():
    data=inputs();request=source_pose_recovery(**data)
    assert request['reference_view']==list(data['reference'].view)
    assert request['reference_frame_id']==2 and request['frame_id']==9
    assert request['maximum_recovery_observations']==1 and request['remaining_real_observations']==1
    assert request['registration_verified'] is False and request['acquires_new_observation'] is True


def ranked_inputs():
    from view2grasp.geometry import unproject
    data=inputs();data['current']=replace(data['current'],timestamp=3.,q=np.full(7,.2))
    alternate=replace(data['reference'],frame_id=4,view=(60.,30.,1.3))
    pixels=np.array([[x,y] for y in range(10,22) for x in range(10,22)])
    points=unproject(pixels,np.ones(len(pixels)),alternate.K,alternate.T_wc)
    return data,alternate,points


def test_source_recovery_ranks_initial_cameras_at_current_robot_pose_without_new_scene_rgbd():
    data,alternate,points=ranked_inputs();calls=[];restored=[]
    class Robot:
        def depth(self,obs):
            assert np.array_equal(obs.q,data['current'].q)
            assert np.array_equal(obs.gripper_q,data['current'].gripper_q)
            calls.append(obs.frame_id)
            return np.full_like(obs.depth,10. if obs.frame_id==4 else .5)
        def set_q(self,q,gripper):restored.append((q.copy(),gripper.copy()))
    before=points.copy()
    request=source_pose_recovery(**data,robot=Robot(),source_candidates=(alternate,),source_points=points)
    assert request['reference_frame_id']==4 and request['reference_view']==list(alternate.view)
    assert request['maximum_recovery_observations']==1 and calls==[2,4]
    assert np.array_equal(points,before) and np.array_equal(restored[-1][0],data['current'].q)
    ranking=request['view_ranking']
    assert ranking['new_scene_observations']==0 and ranking['robot_state_frame_id']==9
    assert ranking['ranked_cameras'][0]['predicted_visible_reference_points']==len(points)
    data['acquired_views'].append(alternate.view)
    assert source_pose_recovery(**data,robot=Robot(),source_candidates=(alternate,),source_points=points) is None
    # No source image is consumed for an unrelated failure, even with candidates.
    data['failure']='invalid_input';calls.clear()
    assert source_pose_recovery(**data,robot=Robot(),source_candidates=(alternate,),source_points=points) is None
    assert not calls


def test_source_ranking_rejects_noninitial_cameras_and_restores_current_model_on_error():
    from view2grasp.postclose_preview import rank_source_pose_views
    data,alternate,points=ranked_inputs();restored=[]
    class Robot:
        def depth(self,obs):raise RuntimeError('self-depth unavailable')
        def set_q(self,q,gripper):restored.append(q.copy())
    future=replace(alternate,frame_id=data['current'].frame_id)
    with pytest.raises(ValueError,match='earlier camera'):
        rank_source_pose_views(Robot(),[future],data['current'],points,[])
    with pytest.raises(RuntimeError,match='self-depth unavailable'):
        rank_source_pose_views(Robot(),[alternate],data['current'],points,[])
    assert len(restored)==2 and all(np.array_equal(q,data['current'].q) for q in restored)
    with pytest.raises(ValueError,match='bounded cameras'):
        rank_source_pose_views(Robot(),[alternate]*5,data['current'],points,[])


@pytest.mark.parametrize('fault',['used','budget','boolean_budget','missing','unknown','clear',
    'unrelated_failure','same_current','already_acquired','nonfinite'])
def test_no_revisit_without_unresolved_pose_and_a_distinct_budgeted_source_view(fault):
    data=inputs()
    if fault=='used':data['used']=True
    elif fault=='budget':data['remaining']=0
    elif fault=='boolean_budget':data['remaining']=True
    elif fault=='missing':data['reference']=None
    elif fault=='unknown':data['preview']['mapping_evidence_available']=True
    elif fault=='clear':data['preview']['occupied_unresolved']=False
    elif fault=='unrelated_failure':data['failure']='invalid_input'
    elif fault=='same_current':data['current']=replace(data['current'],view=(195.,15.,.8))
    elif fault=='already_acquired':data['acquired_views'].append((195.,15.,.8))
    else:data['reference']=replace(data['reference'],view=(np.nan,15.,.8))
    assert source_pose_recovery(**data) is None


@pytest.mark.parametrize('accepted',[False,True])
def test_runner_one_current_source_view_never_retires_history_without_registration(tmp_path,monkeypatch,accepted):
    result,e=run_lifecycle(tmp_path,monkeypatch,initialize_reference=True,max_observations=5,
        pose_recovery=True,registration_ok=False,recovery_registration_ok=accepted)
    assert result['observations']==4 and len(e.camera_requests)==1
    assert e.camera_requests[0]['reference_frame_id']==2
    assert e.camera_requests[0]['registration_failure']=='ambiguous_local_pose'
    assert [f for f,_ in e.annotations]==[3,4]
    assert [f for f,_ in e.retired]==([1,3,4] if accepted else [3,4])
    assert (e.planner.registered_payload_local is not None)==accepted
    assert not e.map.contact_allowed(e.predicted).any()
    assert _contains(e.map.surface,e.neighbor[None])
    assert not result['success']


def test_runner_does_not_exceed_episode_budget_for_pose_verification(tmp_path,monkeypatch):
    result,e=run_lifecycle(tmp_path,monkeypatch,initialize_reference=True,max_observations=5,
        pose_recovery=True,registration_ok=False,pose_recovery_at=5)
    assert result['observations']==5 and not e.camera_requests
    assert [f for f,_ in e.retired]==[3,4,5]


@pytest.mark.parametrize('budget,start,expected',[(5,4,1),(6,3,2)])
def test_opt_in_second_revisit_is_distinct_and_never_bypasses_failed_registration(tmp_path,monkeypatch,budget,start,expected):
    result,e=run_lifecycle(tmp_path,monkeypatch,initialize_reference=True,max_observations=budget,
        pose_recovery=True,registration_ok=False,source_pose_views=2,pose_recovery_at=start)
    assert len(e.camera_requests)==expected
    assert [r['reference_frame_id'] for r in e.camera_requests]==[2,1][:expected]
    assert [r['recovery_observations_used'] for r in e.camera_requests]==list(range(expected))
    assert all(r['maximum_recovery_observations']==2 for r in e.camera_requests)
    assert result['observations']==start+expected<=budget
    assert [f for f,_ in e.retired]==list(range(3,start+expected+1))
    assert e.planner.registered_payload_local is None
    assert not e.map.contact_allowed(e.predicted).any()
    assert _contains(e.map.surface,e.neighbor[None]) and not result['success']


@pytest.mark.parametrize('used,maximum,allowed',[(0,2,True),(1,2,True),(2,2,False),
    (1,1,False),(-1,2,False),(0,3,False),(0,True,False)])
def test_source_recovery_cap_is_explicit_and_bounded(used,maximum,allowed):
    data=inputs();data['used']=used
    assert (source_pose_recovery(**data,maximum_views=maximum) is not None)==allowed


def test_runner_unrelated_registration_failure_does_not_consume_another_view(tmp_path,monkeypatch):
    result,e=run_lifecycle(tmp_path,monkeypatch,initialize_reference=True,max_observations=5,
        pose_recovery=True,registration_ok=False,recovery_failure='invalid_input')
    assert result['observations']==3 and not e.camera_requests


def transaction():
    data=inputs();request=source_pose_recovery(**data)
    return data['current'],{'source_pose_verification':request,
        'coverage_advice':{'purpose':'source_pose_verification','views':[request['reference_view']]}}


def test_real_vlm_contract_requires_the_offered_source_view_and_keeps_repair_budget(tmp_path):
    obs,tx=transaction();before=copy.deepcopy(tx)
    wrong=AgentDecision(action='observe',frame_id=obs.frame_id,camera_id=obs.camera_id,azimuth=0.,elevation=10.,distance=1.3)
    right=wrong.model_copy(update={'azimuth':-165.,'elevation':15.,'distance':.8})
    requests=[];client=client_with_responses(tmp_path,[wrong,right],requests)
    try:answer=client.decide(obs,'Verify source pose',[tx],force_observe=True)
    finally:client.client.close()
    assert answer.azimuth==-165. and client.calls==2
    context=json.loads(requests[0]['messages'][1]['content'][0]['text'])
    assert 'source_pose_verification_context' in context and 'mapping_context' not in context
    assert 'source-pose verification' in context['response_contract']
    assert 'exactly the offered reference_view' in requests[1]['messages'][-1]['content']
    assert tx==before


@pytest.mark.parametrize('used,maximum,allowed',[(1,2,True),(2,2,False),(0,3,False),
    (True,2,False),(-1,2,False)])
def test_vlm_source_context_enforces_second_revisit_cap(used,maximum,allowed):
    from view2grasp.vlm import _source_pose_verification_context
    obs,tx=transaction()
    tx['source_pose_verification'].update(recovery_observations_used=used,maximum_recovery_observations=maximum)
    assert (_source_pose_verification_context(obs,[tx]) is not None)==allowed


@pytest.mark.parametrize('fault',['stale','later_event','verified','purpose','budget','views','unforced','annotation'])
def test_source_pose_context_cannot_escape_its_current_forced_transaction(tmp_path,fault):
    obs,tx=transaction();history=[tx];kwargs={'force_observe':True}
    if fault=='stale':tx['source_pose_verification']['frame_id']=0
    elif fault=='later_event':history.append({'feedback':'new transaction'})
    elif fault=='verified':tx['source_pose_verification']['registration_verified']=True
    elif fault=='purpose':tx['coverage_advice']['purpose']='mapping_only'
    elif fault=='budget':tx['source_pose_verification']['remaining_real_observations']=0
    elif fault=='views':tx['coverage_advice']['views']=[]
    elif fault=='unforced':kwargs={}
    else:kwargs['observe_purpose']='annotation'
    response=AgentDecision(action='observe',frame_id=obs.frame_id,azimuth=0.,elevation=10.,distance=1.3)
    requests=[];client=client_with_responses(tmp_path,[response],requests)
    try:client.decide(obs,'Observe',history,**kwargs)
    finally:client.client.close()
    context=json.loads(requests[0]['messages'][1]['content'][0]['text'])
    assert 'source_pose_verification_context' not in context
