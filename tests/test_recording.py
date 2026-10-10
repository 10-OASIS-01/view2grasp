def test_scene_snapshot_survives_independent_robot_model_compilation():
    from view2grasp.sim import Simulator
    from view2grasp.planning import RobotModel
    import mujoco
    sim=Simulator('block',3,size=128)
    robot=None
    try:
        before=sim.scene_xml
        robot=RobotModel(sim.robot_xml,sim.joint_names,sim.eef_name)
        restored=mujoco.MjModel.from_xml_string(before)
        assert restored.nq==sim.env.sim.model.nq
        assert restored.ngeom==sim.env.sim.model.ngeom
        assert restored.camera('overview').id>=0
        assert 'cube' in before and sim.scene_xml==before
    finally:
        if robot:robot.close()
        sim.close()
