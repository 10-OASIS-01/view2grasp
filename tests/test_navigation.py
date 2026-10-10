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
