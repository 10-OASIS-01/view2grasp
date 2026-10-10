"""Trusted RoboCasa I/O. Only allowlisted observations cross into policy code."""
import os
import sys
import numpy as np
from .types import Observation

from .mobile_cameras import CAMERAS,mounted_camera_configs
TASKS=('NavigateKitchen','DeliverCanToBowl','DeliverBrewedCoffee')


class RoboCasaSimulator:
    def __init__(self,task='NavigateKitchen',seed=0,size=512,layout=1,style=1):
        if task not in TASKS:raise ValueError('unsupported RoboCasa task')
        os.environ.setdefault('MUJOCO_GL','cgl' if sys.platform=='darwin' else 'egl')
        import robocasa
        import robosuite
        if task == 'DeliverCanToBowl':
            # Importing the class registers it with RoboCasa / robosuite's
            # environment metaclass inside this isolated interpreter.
            from . import robocasa_tasks  # noqa: F401
        from robocasa.utils import camera_utils
        # Sensor extrinsics are fixed in a robot body, configured identically
        # for all observation policies. No unseen image is exported to the VLM.
        camera_utils.CAM_CONFIGS.setdefault('PandaOmron',{}).update(mounted_camera_configs())
        from robosuite.controllers import load_composite_controller_config
        config=load_composite_controller_config(robot='PandaOmron')
        parts=config['body_parts']
        arms=parts.get('arms',parts)
        grip=arms['right']['gripper']
        arms['right']=dict(type='JOINT_POSITION',input_type='delta',kp=200,
            damping_ratio=1,impedance_mode='fixed',input_max=1,input_min=-1,
            output_max=.1,output_min=-.1,interpolation=None,gripper=grip)
        self.env=robosuite.make(task,robots='PandaOmron',controller_configs=config,
            has_renderer=False,has_offscreen_renderer=True,use_camera_obs=False,
            use_object_obs=False,camera_names=list(CAMERAS),camera_heights=size,camera_widths=size,
            camera_depths=True,control_freq=20,ignore_done=True,
            seed=seed,layout_ids=[layout],style_ids=[style],generative_textures=None,
            randomize_cameras=False,initialization_noise=None)
        self.env.reset()
        # Match the robot-only depth renderer: multisample edge blending is not
        # a metric depth measurement and corrupts self-occlusion subtraction.
        self.env.sim.model.vis.quality.offsamples=0
        self.scene_xml=self.env.sim.model.get_xml()
        self.task=task;self.seed=seed;self.size=size;self.frame_id=0;self.phase='observe'
        self.robot=self.env.robots[0]
        self.controller=self.robot.composite_controller
        self.action_layout=dict(self.controller._action_split_indexes)
        expected={'right':7,'right_gripper':1,'base':3,'torso':1}
        for name,width in expected.items():
            if name not in self.action_layout or self.action_layout[name][1]-self.action_layout[name][0]!=width:
                raise RuntimeError(f'unvalidated mobile action layout: {self.action_layout}')
        self.joint_names=list(self.robot.robot_model.arm_joints)
        self.qidx=[self.env.sim.model.get_joint_qpos_addr(n) for n in self.joint_names]
        self.gripidx=list(self.robot._ref_gripper_joint_pos_indexes['right'])
        self.aux_names=list(self.robot.robot_model.base_joints)+list(self.robot.robot_model.torso_joints)
        self.robot_xml=self.robot.robot_model.get_xml()
        self.eef_name=self.robot.gripper['right'].important_sites['grip_site']
        self.base_name=self.robot.robot_model.base.naming_prefix+'base'
        from .mobile_control import MobileVelocityAdapter
        self.base_driver=MobileVelocityAdapter(self.robot,self.env.sim,self.base_name)
        self.arm_step_limit=.1
        self.camera=CAMERAS[0]
        self.frames=[];self.states=[];self.actions=[]
        # Export ONLY task language, not the remaining episode metadata.
        self.instruction=str(self.env.get_ep_meta()['lang'])
        self.initial_arm=self.q.copy()
        self.torso_hold=self.torso_q.copy()
        for _ in range(10):self.step(self.q,-1,record=False)

    @property
    def q(self):return self.env.sim.data.qpos[self.qidx].copy()

    @property
    def time(self):return float(self.env.sim.data.time)

    @property
    def gripper_q(self):return self.env.sim.data.qpos[self.gripidx].copy()

    @property
    def torso_q(self):
        return np.array([self.env.sim.data.get_joint_qpos(n) for n in self.robot.robot_model.torso_joints])

    @property
    def base_pose(self):
        p=self.env.sim.data.get_body_xpos(self.base_name)
        R=self.env.sim.data.get_body_xmat(self.base_name)
        return np.array([p[0],p[1],np.arctan2(R[1,0],R[0,0])])

    def auxiliary_state(self):
        return {n:float(self.env.sim.data.get_joint_qpos(n)) for n in self.aux_names}

    def observe(self,camera=None):
        from robosuite.utils.camera_utils import get_camera_intrinsic_matrix,get_camera_extrinsic_matrix,get_real_depth_map
        camera=camera or self.camera
        if camera not in CAMERAS:raise ValueError('only mounted cameras are allowed')
        self.camera=camera
        self.env.sim._render_context_offscreen.gl_ctx.make_current()
        rgb,depth=self.env.sim.render(width=self.size,height=self.size,camera_name=camera,depth=True)
        K=get_camera_intrinsic_matrix(self.env.sim,camera,self.size,self.size);K[:2,2]-=.5
        self.frame_id+=1
        return Observation(self.frame_id,rgb[::-1].copy(),get_real_depth_map(self.env.sim,depth[::-1]).copy(),
            K,get_camera_extrinsic_matrix(self.env.sim,camera),self.q,self.gripper_q,
            float(self.env.sim.data.time),(0.,0.,0.),camera,self.base_pose,self.torso_q)

    def camera_calibration(self,camera):
        """Return current mounted-camera calibration without acquiring an image."""
        if camera not in CAMERAS:raise ValueError('only mounted cameras are allowed')
        from robosuite.utils.camera_utils import get_camera_intrinsic_matrix,get_camera_extrinsic_matrix
        K=get_camera_intrinsic_matrix(self.env.sim,camera,self.size,self.size);K[:2,2]-=.5
        return K,get_camera_extrinsic_matrix(self.env.sim,camera)

    def step(self,q,gripper,base_velocity=None,record=True):
        from .mobile_control import absolute_joint_target_action
        velocity=np.zeros(3) if base_velocity is None else np.asarray(base_velocity,dtype=float)
        if not np.isfinite(velocity).all() or np.linalg.norm(velocity[:2])>.20001 or abs(velocity[2])>.40001:
            raise ValueError('mobile velocity exceeds declared limits')
        arm_action=absolute_joint_target_action(self.q,q,self.arm_step_limit)
        action=self.controller.create_action_vector({'right':arm_action,'right_gripper':np.array([gripper]),
            'base':self.base_driver.action(velocity),'torso':np.zeros(1),'base_mode':1 if np.any(velocity) else -1})
        # No rewards, observations, task done flags or info return to the caller.
        self.env.step(action)
        if record:
            self.actions.append(action.copy());self.states.append(self.env.sim.get_state().flatten())
            if len(self.actions)%2==0:
                rgb=self.env.sim.render(width=640,height=480,camera_name=self.camera)[::-1].copy()
                self.frames.append((rgb,self.phase,float(self.env.sim.data.time)))

    def close(self):self.env.close()


def installation_report():
    import importlib.metadata
    from pathlib import Path
    import robocasa
    assets=Path(robocasa.__file__).parent/'models/assets'
    required=('textures','fixtures','objects')
    ready={name: (assets/name).is_dir() and any((assets/name).rglob('*.xml' if name!='textures' else '*.png')) for name in required}
    return {'versions':{p:importlib.metadata.version(p) for p in ('robocasa','robosuite','mujoco','numpy')},
            'assets':ready,'assets_ready':all(ready.values()),'assets_path':str(assets)}
