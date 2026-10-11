import numpy as np
from view2grasp.navigation import SE2Planner,velocity_command,transform_body,NavigationCollision
from view2grasp.mapping import DepthMap


def test_se2_checks_edge_middle_and_rotation():
    p=SE2Planner(lambda q:not (.04<q[0]<.06))
    assert not p.edge(np.zeros(3),np.array([.1,0,0]))
    p=SE2Planner(lambda q:abs(q[2])<.2)
    assert not p.edge(np.zeros(3),np.array([0,0,.4]))


def test_path_routes_around_obstacle_and_hints_do_not_override_collision():
    valid=lambda q: not (.04<q[0]<.08 and abs(q[1])<.04)
    p=SE2Planner(valid)
    route=p.plan([0,0,0],[.12,0,0],hints=[[0,0],[.12,0]],time_budget=2)
    assert route is not None and np.max(np.abs(route[:,1]))>=.04
    assert all(p.edge(a,b) for a,b in zip(route,route[1:]))


def test_unknown_full_robot_and_payload_block_navigation():
    from types import SimpleNamespace
    floor=np.array([[0,0,0]])
    body=np.array([[0.,0.,.01],[0,0,.5]])
    c=NavigationCollision(DepthMap(workspace_bounds=None),body,floor,0)
    assert not c([0,0,0])
    observed=SimpleNamespace(classify=lambda p:np.where(p[:,0]>.1,2,1))
    assert NavigationCollision(observed,body,floor,0)([0,0,0])
    assert not NavigationCollision(observed,body,floor,0,payload=np.array([[.2,0,.8]]))([0,0,0])


def test_local_self_shadow_certificate_is_bounded_and_never_waives_occupied_or_floor():
    from types import SimpleNamespace
    floor=np.array([[x,0,0] for x in np.linspace(-.2,.2,20)])
    body=np.array([[0.,0.,.01],[0,0,.5]])
    unknown=SimpleNamespace(classify=lambda p:np.zeros(len(p),dtype=np.uint8))
    c=NavigationCollision(unknown,body,floor,0,shadow_pose=np.zeros(3),shadow_distance=.03)
    assert c([.02,0,0])
    assert not c([.04,0,0])
    occupied=SimpleNamespace(classify=lambda p:np.full(len(p),2,dtype=np.uint8))
    c=NavigationCollision(occupied,body,floor,0,shadow_pose=np.zeros(3),shadow_distance=.03)
    assert not c([.01,0,0])


def test_velocity_limits_and_body_rotation():
    command=velocity_command(np.array([0,0,np.pi/2]),np.array([1,0,0]))
    assert np.isclose(command[0],0) and command[1]<0
    assert np.linalg.norm(command[:2])<=.2+1e-9 and abs(command[2])<=.4
    assert np.allclose(transform_body([[1,0,1]],[1,2,np.pi/2]),[[1,3,1]])


def test_occupied_payload_diagnostic_links_depth_evidence_without_claiming_source():
    import json
    from scipy.spatial import cKDTree
    from view2grasp.types import Observation

    observed=DepthMap(workspace_bounds=None)
    observed.surface=np.array([[0.,0.,.5]])
    observed.tree=cKDTree(observed.surface)
    K=np.array([[10.,0,2],[0,10,2],[0,0,1]])
    for frame_id,depth in ((8,.5),(9,2.)):
        obs=Observation(frame_id,np.zeros((5,5,3),dtype=np.uint8),np.full((5,5),depth),
                        K,np.eye(4),np.zeros(7),np.zeros(2),0.,(0,0,0),camera_id='robot0_nav_front')
        observed.views.append(obs)
        observed.self_masks.append(np.zeros((5,5),dtype=bool))
        observed.occlusion_masks.append(np.full((5,5),frame_id==9,dtype=bool))
    body=np.array([[0.,0.,.01],[.2,0,.5]])
    c=NavigationCollision(observed,body,[[0,0,0]],0,payload=[[0,0,.5]],
                          shadow_pose=[0,0,0],shadow_distance=.03)
    assert not c([0,0,0])  # Occupied payload is never waived by self-shadow.
    report=c.diagnostic()
    json.dumps(report,allow_nan=False)
    failure=report['last_rejection']
    assert failure['reason']=='occupied robot sweep'
    assert failure['state_counts']=={'unknown':1,'free':1,'occupied':1}
    sample=failure['samples'][0]
    assert sample['component']=='payload' and sample['sample_index']==2
    assert sample['world']==[0,0,.5] and sample['state']=='occupied'
    assert sample['nearest_occupied_endpoint']=={'world':[0.,0.,.5],'distance_m':0.}
    assert [v['frame_id'] for v in sample['depth_evidence']]==[8,9]
    assert [v['relation'] for v in sample['depth_evidence']]==['at_or_behind_depth_return','robot_occluded']
    assert 'source_unavailable' in report['map_context']['source_attribution']
    assert len(observed.views)==2 and np.array_equal(observed.surface,[[0,0,.5]])
    assert not c([0,0,0])  # Reading diagnostics cannot turn a collision into free space.


def test_unknown_and_floor_diagnostics_remain_bounded_and_explain_certificates():
    import json
    from types import SimpleNamespace

    unknown=SimpleNamespace(classify=lambda p:np.zeros(len(p),dtype=np.uint8))
    body=np.vstack([[[0,0,.01]],np.column_stack([np.zeros(80),np.zeros(80),np.linspace(.1,1,80)])])
    c=NavigationCollision(unknown,body,[[0,0,0],[.04,0,0]],0,
                          shadow_pose=[0,0,0],shadow_distance=.03)
    assert not c([.04,0,0])
    failure=c.diagnostic()['last_rejection']
    assert failure['shadow_distance_m']==.03
    assert failure['rejected_sample_count']==80 and failure['samples_truncated']
    assert len(failure['samples'])==32
    assert all(s['state']=='unknown' for s in failure['samples'])
    assert failure['bounds_world']==[[.04,0,.1],[.04,0,1.]]

    assert not c([.1,0,0])
    report=c.diagnostic()
    assert report['last_rejection']['reason']=='unknown floor support'
    sample=report['last_rejection']['samples'][0]
    assert np.isclose(sample['nearest_floor_support']['distance_m'],.06)
    assert sample['nearest_floor_support']['xy']==[.04,0.]
    json.dumps(report,allow_nan=False)


def test_planner_records_rejected_edge_and_resets_previous_attempt():
    from types import SimpleNamespace

    # Clear endpoints with a collision halfway along the controller's segment.
    observed=SimpleNamespace(classify=lambda p:np.where((p[:,0]>.04)&(p[:,0]<.06),2,1))
    body=np.array([[0.,0.,.01],[0,0,.5]])
    floor=np.array([[x,0,0] for x in np.linspace(0,.2,50)])
    collision=NavigationCollision(observed,body,floor,0)
    planner=SE2Planner(collision)
    assert not planner.edge([0.,0,0],[.1,0,0])
    report=planner.diagnostic()
    edge=report['last_rejected_edge']
    assert edge['start']==[0,0,0] and edge['end']==[.1,0,0]
    assert .4<=edge['fraction']<.6
    assert edge['pose']==report['collision']['last_rejection']['pose']
    # A successful validity check must not destroy evidence of the rejected edge.
    assert collision([0,0,0])
    assert planner.diagnostic()['collision']['last_rejection']['reason']=='occupied robot sweep'
    assert planner.plan([.1,0,0],[.12,0,0]) is not None
    assert planner.diagnostic()['last_rejected_edge'] is None
    assert planner.diagnostic()['collision']['last_rejection'] is None
    assert planner.last_failure==''


def test_missing_floor_diagnostic_is_json_safe_and_still_blocks():
    import json
    from types import SimpleNamespace

    observed=SimpleNamespace(classify=lambda p:np.ones(len(p),dtype=np.uint8))
    c=NavigationCollision(observed,[[0,0,.01],[0,0,.5]],[],0)
    assert not c([0,0,0])
    report=c.diagnostic()
    assert report['last_rejection']['samples'][0]['nearest_floor_support'] is None
    json.dumps(report,allow_nan=False)
