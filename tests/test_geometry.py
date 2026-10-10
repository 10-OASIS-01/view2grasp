import numpy as np
import pytest
from view2grasp.geometry import unproject,project,pixel_point
from view2grasp.types import AgentDecision,Observation

def test_lowered_virtual_camera_view_is_within_agent_contract():
    d=AgentDecision(action='observe',frame_id=1,elevation=-10,distance=.8)
    assert d.elevation==-10

def observation():
    return Observation(1,np.zeros((32,32,3),dtype=np.uint8),np.ones((32,32)),
        np.array([[30,0,16],[0,30,16],[0,0,1]]),np.eye(4),np.zeros(7),np.array([.04,-.04]),0,(0,55,.8))

def test_camera_round_trip():
    obs=observation(); uv=np.array([[2.3,4.2],[23,29],[16,16]])
    for yaw in (0,.3,1.5):
        from scipy.spatial.transform import Rotation
        T=np.eye(4); T[:3,:3]=Rotation.from_euler('xyz',[.2,.6,yaw]).as_matrix(); T[:3,3]=[.3,-.5,1.2]
        p=unproject(uv,[.5,1,2],obs.K,T); actual,z=project(p,obs.K,T)
        assert np.max(abs(actual-uv))<1e-9
        assert np.allclose(z,[.5,1,2])

@pytest.mark.parametrize('point',[(-.1,.5),(.5,1.1),(float('nan'),.5)])
def test_invalid_grounding(point):
    with pytest.raises(ValueError):
        AgentDecision(action='grasp',frame_id=1,point=point,bbox=(0,0,1,1),part_bbox=(0,0,1,1))

def test_depth_edge_rejected():
    obs=observation(); obs.depth[16,16]=2
    with pytest.raises(ValueError,match='edge'): pixel_point(obs,(.5,.5))

def test_frame_boundary_no_simulator():
    from dataclasses import fields
    assert set(f.name for f in fields(Observation))=={'frame_id','rgb','depth','K','T_wc','q','gripper_q','timestamp','view','camera_id','base_pose','torso_q'}
    import inspect
    import view2grasp.planning as planning
    import view2grasp.geometry as geometry
    for module in (planning,geometry):
        source=inspect.getsource(module)
        assert 'from .sim import' not in source
        assert 'body_xpos' not in source
        assert 'evaluation_truth' not in source

def test_grasp_cone_rotations_and_surface_offset():
    from view2grasp.geometry import grasp_candidates
    obs=observation();obs.K[0,0]=obs.K[1,1]=300
    d=AgentDecision(action='grasp',frame_id=1,point=(.5,.5),bbox=(.3,.3,.7,.7),part_bbox=(.3,.3,.7,.7))
    candidates,_=grasp_candidates(obs,d,.5,np.eye(3))
    assert len(candidates)>20
    p=pixel_point(obs,d.point);ray=p/np.linalg.norm(p)
    for g in candidates:
        assert np.rad2deg(np.arccos(np.clip(g.approach@ray,-1,1)))<=15.000001
        assert np.allclose(g.rotation.T@g.rotation,np.eye(3),atol=1e-8)
        assert abs(np.linalg.det(g.rotation)-1)<1e-8
        assert any(abs(np.linalg.norm(g.position-p)-x)<1e-8 for x in (.008,.016,.024))
    with pytest.raises(ValueError,match='stale'):
        grasp_candidates(obs,d.model_copy(update={'frame_id':0}),.5)

def test_upward_leading_shell_keeps_highest_point_per_xy_voxel():
    from view2grasp.runner import leading_shell
    points=np.array([[.001,.001,.8],[.002,.001,.9],[.021,.0,.85]])
    shell=leading_shell(points,.005)
    assert len(shell)==2
    assert .9 in shell[:,2]
    assert .8 not in shell[:,2]
