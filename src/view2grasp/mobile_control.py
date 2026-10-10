"""Trusted proprioceptive velocity adapter for the pinned PandaOmron controller."""
import numpy as np
import mujoco


def absolute_joint_target_action(current, target, max_step=.1):
    """Convert a radian joint target to robosuite's normalized delta action.

    RoboCasa sends zero actions while resetting.  Keeping the native delta
    semantics makes those actions hold the configured initial pose; switching
    the controller to absolute semantics would instead drive every joint to
    zero before the episode begins.
    """
    current=np.asarray(current,dtype=float);target=np.asarray(target,dtype=float)
    if current.shape!=target.shape or current.ndim!=1:raise ValueError('joint target shape mismatch')
    if max_step<=0 or not np.isfinite(max_step):raise ValueError('invalid joint step limit')
    if not np.isfinite(current).all() or not np.isfinite(target).all():raise ValueError('nonfinite joint target')
    return np.clip((target-current)/max_step,-1.,1.)


class MobileVelocityAdapter:
    def __init__(self, robot, sim, base_name):
        self.robot=robot;self.sim=sim
        self.controller=robot.composite_controller.part_controllers['base']
        self.bid=sim.model.body_name2id(base_name)
        joints=[sim.model.joint_name2id(n) for n in robot.robot_model.base_joints]
        self.dofs=np.asarray(sim.model.jnt_dofadr[joints])
        self.actuators=np.asarray(robot._ref_actuators_indexes_dict['base'])
        self.low=sim.model.actuator_ctrlrange[self.actuators,0]
        self.high=sim.model.actuator_ctrlrange[self.actuators,1]
        self.gain=sim.model.actuator_gainprm[self.actuators,0]
        self.friction=sim.model.dof_frictionloss[self.dofs]
        if len(self.dofs)!=3 or np.any(self.gain<=0):raise ValueError('unvalidated mobile actuator model')

    def action(self, body_velocity):
        velocity=np.asarray(body_velocity,dtype=float)
        if not np.any(velocity):return np.zeros(3)
        model=self.sim.model._model;data=self.sim.data._data
        jacp=np.zeros((3,model.nv));jacr=np.zeros_like(jacp)
        mujoco.mj_jacBody(model,data,jacp,jacr,self.bid)
        R=self.sim.data.body_xmat[self.bid].reshape(3,3)
        world=R[:2,:2]@velocity[:2]
        J=np.vstack([jacp[:2,self.dofs],jacr[2:3,self.dofs]])
        desired=np.linalg.solve(J,np.r_[world,velocity[2]])
        # Native velocity actuators otherwise stall below the model's Coulomb
        # friction threshold. This is model-based feedforward, not a physics edit.
        setpoint=desired+np.sign(desired)*self.friction/self.gain
        normalized=(setpoint-(self.high+self.low)/2)/((self.high-self.low)/2)
        _,current=self.controller.get_base_pose()
        initial=self.controller.init_ori
        theta=np.arctan2(current[1,0],current[0,0])-np.arctan2(initial[1,0],initial[0,0])
        c,s=np.cos(theta),np.sin(theta)
        normalized[:2]=np.array([[c,-s],[s,c]])@normalized[:2]
        if np.max(np.abs(normalized))>1:raise ValueError('requested velocity exceeds controller range')
        return normalized
