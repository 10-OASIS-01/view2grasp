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


def navigation_region_scene(kind='box',role='navigation_goal'):
    obs=observation()
    # Downward calibrated camera: depth 1 is an elevated surface, depth 2 is
    # floor at world z=0. The generic region center is on the elevated surface.
    obs.T_wc[:3,:3]=np.diag([1.,-1.,-1.]);obs.T_wc[2,3]=2.
    obs.depth[10:24,21:28]=2.
    obs.depth[10,21]=np.nan;obs.depth[11,21]=0.
    region={'box':(.1,.1,.9,.9)} if kind=='box' else {
        'points':[(.1,.1),(.9,.1),(.9,.9),(.1,.9)]}
    return obs,GroundingProposal(kind=kind,role=role,frame_id=1,**region)


@pytest.mark.parametrize('kind',['box','polygon'])
def test_navigation_region_selects_observed_floor_before_its_interior(kind):
    from view2grasp.geometry import project
    from view2grasp.grounding import proposal_mask
    obs,proposal=navigation_region_scene(kind)
    assert grounded_world(obs,proposal)[2]==pytest.approx(1.)
    world=grounded_world(obs,proposal,floor_z=0.)
    assert world[2]==pytest.approx(0.)
    uv,_=project(world[None],obs.K,obs.T_wc);u,v=np.rint(uv[0]).astype(int)
    assert proposal_mask(proposal,obs)[v,u]
    assert 21<=u<28 and 10<=v<24 and np.isfinite(obs.depth[v,u])


@pytest.mark.parametrize('kind',['box','polygon'])
def test_navigation_region_without_floor_does_not_borrow_floor_outside_it(kind):
    obs,proposal=navigation_region_scene(kind)
    obs.depth[:]=1.;obs.depth[:2,:2]=2.
    with pytest.raises(ValueError,match='navigation region has no observed floor'):
        grounded_world(obs,proposal,floor_z=0.)


def test_navigation_point_is_not_moved_to_nearby_floor():
    obs,_=navigation_region_scene()
    proposal=GroundingProposal(kind='points',role='navigation_goal',frame_id=1,points=[(.5,.5)])
    with pytest.raises(ValueError,match='navigation goal is not on observed floor'):
        grounded_world(obs,proposal,floor_z=0.)
    obs.depth[16,16]=1.941
    assert grounded_world(obs,proposal,floor_z=0.)[2]==pytest.approx(.059)
    obs.depth[16,16]=1.939
    with pytest.raises(ValueError,match='navigation goal is not on observed floor'):
        grounded_world(obs,proposal,floor_z=0.)


@pytest.mark.parametrize('change',[{'frame_id':2},{'camera_id':'wrist'}])
def test_navigation_floor_filter_preserves_frame_and_camera_validation(change):
    obs,proposal=navigation_region_scene()
    with pytest.raises(ValueError,match='stale grounding'):
        grounded_world(replace(obs,**change),proposal,floor_z=0.)


@pytest.mark.parametrize('role',['object','place_region','look_at'])
def test_non_navigation_region_keeps_generic_grounding_semantics(role):
    obs,proposal=navigation_region_scene(role=role)
    original=grounded_world(obs,proposal)
    assert original[2]==pytest.approx(1.)
    assert np.array_equal(grounded_world(obs,proposal,floor_z=0.),original)


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
