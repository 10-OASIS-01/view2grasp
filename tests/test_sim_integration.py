"""Actual renderer/robot tests; no VLM request is made by pytest."""
import numpy as np
import pytest

@pytest.fixture(scope="module")
def pair():
    from view2grasp.sim import Simulator
    from view2grasp.planning import RobotModel
    sim=Simulator("block",7,256)
    robot=RobotModel(sim.robot_xml,sim.joint_names,sim.eef_name)
    yield sim,robot
    robot.close(); sim.close()

def test_independent_robot_fk(pair):
    sim,robot=pair
    p,R=robot.pose(sim.q)
    assert np.allclose(p,sim.env.sim.data.get_site_xpos(sim.eef_name),atol=1e-8)
    assert np.allclose(R,sim.env.sim.data.get_site_xmat(sim.eef_name),atol=1e-8)

def test_renderer_context_switch_does_not_corrupt_depth(pair):
    sim,robot=pair
    for view in [(0,55,.8),(90,25,1.3),(-90,70,1.0)]:
        sim.move_camera(*view)
        before=sim.observe(); own=robot.depth(before); after=sim.observe()
        assert np.allclose(before.depth,after.depth,atol=1e-6)
        # Driver color rounding may differ by one 8-bit level after a context switch.
        assert np.max(np.abs(before.rgb.astype(int)-after.rgb.astype(int)))<=1
        assert np.sum(np.abs(own-before.depth)<.003)>100

def test_self_depth_has_no_debug_sites_or_spurious_foreground(pair):
    sim,robot=pair
    for view in [(0,20,.55),(0,55,.8),(90,20,.55),(180,30,1.3)]:
        sim.move_camera(*view); obs=sim.observe(); own=robot.depth(obs)
        # Scene objects can occlude the robot, but a robot-only image cannot
        # contain a surface in front of the complete scene's first return.
        assert not np.any((own<3)&(obs.depth>own+.0001))

def test_metric_table_depth_not_just_roundtrip(pair):
    from view2grasp.geometry import cloud,fit_table
    sim,_=pair
    for view in [(0,55,.8),(0,80,1.3)]:
        sim.move_camera(*view);obs=sim.observe()
        table=fit_table(cloud(obs),np.random.default_rng(7))
        # Truth is used ONLY by this test to catch half-pixel calibration errors.
        assert abs(table-.8)<.0003

def test_ik_roundtrip_and_unreachable(pair):
    sim,robot=pair
    p,R=robot.pose(sim.q)
    q=robot.ik(p+[0,0,.015],R,sim.q)
    assert q is not None
    actual,_=robot.pose(q)
    assert np.linalg.norm(actual-(p+[0,0,.015]))<.002
    assert robot.ik([10,10,10],R,sim.q,max_iter=30) is None

def test_robot_xml_excludes_scene(pair):
    sim,robot=pair
    assert 'cube' not in sim.robot_xml
    assert 'table_collision' not in sim.robot_xml
    assert all(not (robot.model.geom(i).name or '').startswith(('cube','occluder','distractor')) for i in range(robot.model.ngeom))

def test_camera_orbit_prediction_uses_only_calibration(pair):
    from view2grasp.geometry import orbit_pose
    sim,_=pair
    for view in [(75,0,1.3),(-135,35,1.1),(0,80,.65)]:
        sim.move_camera(*view)
        assert np.allclose(sim.observe().T_wc,orbit_pose(*view),atol=1e-8)

def test_collision_padding_encloses_original_robot_meshes(pair):
    sim,robot=pair
    for gid,vertices in robot.proxy_vertices.items():
        mid=robot.model.geom_dataid[gid];start=robot.model.mesh_vertadr[mid];n=robot.model.mesh_vertnum[mid]
        original=robot.model.mesh_vert[start:start+n]
        hull=robot.local_hulls[gid]
        assert np.max(original@hull[:,:3].T+hull[:,3])<-.0029
    sim.validate_initial_proxies(robot.points(sim.q,sim.gripper_q)[0])

def test_pad_center_to_control_site_transform(pair):
    sim,robot=pair;robot.set_q(sim.q,np.array([.04,-.04]))
    offset=robot.grasp_center_offset()
    assert np.allclose(offset,[0,0,-.0036],atol=1e-6)
    p,R=robot.pose()
    pad_ids=[gid for gid,_,pad,_ in robot.parts if pad]
    assert np.allclose(p+R@offset,robot.data.geom_xpos[pad_ids].mean(0))

def test_absolute_joint_controller_tracks_radians(pair):
    sim,_=pair
    target=sim.q.copy();target[0]+=.025
    for _ in range(40):sim.step(target,-1,record=False)
    assert np.max(np.abs(sim.q-target))<.008
