import inspect
import numpy as np
import pytest
from dataclasses import replace
from view2grasp.types import GroundingProposal
from view2grasp.mobile_runner import grounded_world,support_plane,payload_envelope,envelope_boundary,next_high_level_camera
from view2grasp.types import AgentDecision
from test_geometry import observation
from view2grasp.mobile_control import absolute_joint_target_action


def test_mobile_and_perception_planners_do_not_import_privileged_adapters():
    from view2grasp import grounding,navigation,geometry,planning,mapping,vlm
    for module in (grounding,navigation,geometry,planning,mapping,vlm):
        source=inspect.getsource(module)
        assert 'from .sim import' not in source
        assert 'from .robocasa_backend import' not in source
        assert 'from .mobile_evaluation import' not in source
        assert 'evaluation_truth' not in source
        assert 'target_fixture' not in source


def test_world_grounding_rejects_wrong_camera_and_invalid_depth():
    obs=observation()
    p=GroundingProposal(kind='points',role='navigation_goal',frame_id=1,points=[(.5,.5)])
    assert np.isclose(grounded_world(obs,p)[2],1)
    with pytest.raises(ValueError,match='stale'):
        grounded_world(replace(obs,camera_id='wrist'),p)
    obs.depth[:]=0
    with pytest.raises(ValueError,match='depth'):grounded_world(obs,p)


def test_support_plane_not_tied_to_tabletop_world_origin():
    points=np.array([[x,y,.82] for x in np.linspace(2,2.4,20) for y in np.linspace(3,3.4,20)])
    height,selected=support_plane(points,anchor=np.array([2.2,3.2,.92]))
    assert abs(height-.82)<1e-9 and len(selected)==400


def test_payload_is_filled_not_only_a_surface():
    pts=np.array([[x,y,z] for x in (0,.04) for y in (0,.04) for z in (0,.04)])
    envelope=payload_envelope(pts)
    assert np.any(np.all((envelope>.005)&(envelope<.035),axis=1))
    boundary=envelope_boundary(envelope)
    assert len(boundary)<len(envelope)
    assert np.min(np.linalg.norm(boundary-np.array([.015,.015,.015]),axis=1))>.001


def test_navigation_cameras_are_fixed_body_mounted_and_look_down():
    from view2grasp.mobile_cameras import mounted_camera_configs
    from scipy.spatial.transform import Rotation
    for cfg in mounted_camera_configs().values():
        assert cfg['parent_body']=='mobilebase0_support'
        R=Rotation.from_quat(np.array(cfg['quat'])[[1,2,3,0]]).as_matrix()
        assert (-R[:,2])[2]<-.4
        assert np.allclose(R.T@R,np.eye(3))


def test_absolute_joint_targets_are_converted_to_bounded_delta_actions():
    current=np.array([0.,-.5,1.])
    target=np.array([.02,-.8,1.2])
    action=absolute_joint_target_action(current,target,.1)
    assert np.allclose(action,[.2,-1.,1.])
    with pytest.raises(ValueError,match='shape'):
        absolute_joint_target_action(current,target[:2])


def test_navigation_reacquires_grounding_camera_not_last_safety_camera():
    rng=np.random.default_rng(1)
    decision=AgentDecision(action='navigate',frame_id=7,camera_id='robot0_nav_right')
    assert next_high_level_camera('active',decision,'robot0_nav_overhead',rng)=='robot0_nav_right'
    decision=AgentDecision(action='observe',frame_id=8,camera_id='robot0_nav_right',
                           observation_camera='robot0_nav_high_front')
    assert next_high_level_camera('active',decision,'robot0_nav_overhead',rng)=='robot0_nav_high_front'
