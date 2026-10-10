"""PandaOmron control/FK smoke in Lift; explicitly NOT a RoboCasa task result."""
import os
import sys
os.environ.setdefault('MUJOCO_GL','cgl' if sys.platform=='darwin' else 'egl')
from pathlib import Path
import json
import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config
from view2grasp.planning import RobotModel


def main():
    config=load_composite_controller_config(robot='PandaOmron')
    parts=config['body_parts'];arms=parts.get('arms',parts)
    grip=arms['right']['gripper']
    arms['right']=dict(type='JOINT_POSITION',input_type='delta',kp=200,damping_ratio=1,
        impedance_mode='fixed',input_max=1,input_min=-1,output_max=.1,output_min=-.1,
        interpolation=None,gripper=grip)
    env=robosuite.make('Lift',robots='PandaOmron',controller_configs=config,
        has_renderer=False,has_offscreen_renderer=True,use_camera_obs=False,use_object_obs=False,
        initialization_noise=None,control_freq=20,seed=3)
    model=None
    try:
        env.reset();r=env.robots[0];c=r.composite_controller
        joints=list(r.robot_model.arm_joints)
        qidx=[env.sim.model.get_joint_qpos_addr(n) for n in joints]
        q=env.sim.data.qpos[qidx].copy()
        auxiliary={n:float(env.sim.data.get_joint_qpos(n)) for n in list(r.robot_model.base_joints)+list(r.robot_model.torso_joints)}
        name=r.gripper['right'].important_sites['grip_site']
        model=RobotModel(r.robot_model.get_xml(),joints,name,include_base=True)
        model.set_auxiliary_state(auxiliary);pred,_=model.pose(q)
        actual=env.sim.data.get_site_xpos(name)
        fk_error=float(np.linalg.norm(pred-actual))
        base=r.robot_model.base.naming_prefix+'base'
        from view2grasp.mobile_control import MobileVelocityAdapter,absolute_joint_target_action
        driver=MobileVelocityAdapter(r,env.sim,base)
        start=env.sim.data.get_body_xpos(base).copy()
        for _ in range(10):
            action=c.create_action_vector({'right':absolute_joint_target_action(env.sim.data.qpos[qidx],q,.1),'right_gripper':[-1],
                'base':driver.action(np.array([.02,0,0])),'torso':[0.],'base_mode':1})
            env.step(action)
        end=env.sim.data.get_body_xpos(base).copy()
        proxy_points=model.points(q)[0]
        finite=bool(np.isfinite(proxy_points).all())
        report={'kind':'robot_control_smoke_in_Lift_NOT_RoboCasa_episode',
                'action_layout':c._action_split_indexes,'arm_joint_names':joints,
                'fk_error_m':fk_error,'collision_samples':len(proxy_points),'finite_collision_proxies':finite,
                'base_displacement_m':(end-start).tolist(),'control_steps':10}
        output=Path(__file__).resolve().parents[1]/'artifacts/mobile-robot-smoke.json'
        output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
        assert finite and fk_error<.001 and np.linalg.norm(end[:2]-start[:2])>.001
    finally:
        if model:model.close()
        env.close()

if __name__=='__main__':main()
