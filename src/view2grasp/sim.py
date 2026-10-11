"""Trusted simulator adapter. Only observe()/proprioception cross the agent boundary."""
import os
os.environ.setdefault("MUJOCO_GL", "cgl" if __import__("sys").platform == "darwin" else "egl")
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import robosuite
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject, CompositeBodyObject
from robosuite.models.tasks import ManipulationTask
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, get_real_depth_map
from .types import Observation

INSTRUCTIONS = {
    "block": "Pick up the red block and hold it above the table.",
    "mug": ("Grasp the mug ONLY by its handle, lift it, and hold it above the table. "
            "Use one active broadside observation, then ground by observation 2 if solid handle material is visible. "
            "From the initial overhead image, estimate the handle's radial direction and choose a view about 90 degrees "
            "to either side of it so the aperture is broadside, preferring the solution on the world -X / azimuth ±180 side "
            "where the Panda base can reach. State the image-based radial estimate and the derived broadside angle in your "
            "reason; looking directly along the handle's radial direction is not broadside. Use elevation about -10 to 30 degrees. "
            "Represent the mug and open aperture with an object proposal (a polygon hole may exclude the aperture). "
            "The grasp_part must isolate the straight outer vertical handle bar, excluding the cup body and both junctions. "
            "Put any contact point at the vertical midpoint of that straight segment, on solid material. Leave clearance "
            "above and below for the complete fingers and hand, not only the fingertip contact."),
    "occluded": "Pick up the blue block and place it at the center of the yellow destination. Avoid the red obstacles.",
}


def make_mug():
    parts = []; positions = []; quats = []
    def add(o,p,q=None):
        parts.append(o); positions.append(p); quats.append(q)
    color = [.16,.62,.73,1]
    add(CylinderObject("bottom",size=[.034,.004],rgba=color,density=350),[0,0,-.091])
    for i, theta in enumerate(np.linspace(0,2*np.pi,20,endpoint=False)):
        add(BoxObject(f"wall{i}",size=[.0055,.0035,.095],rgba=color,density=350),
            [.031*np.cos(theta),.031*np.sin(theta),0],
            [np.cos((theta+np.pi/2)/2),0,0,np.sin((theta+np.pi/2)/2)])
    # A real, open rectangular handle sized for the Panda finger collision
    # bodies and palm envelope: about 122 mm radial and 160 mm vertical clear aperture.
    # The unusually generous aperture is intentional: the project tests true
    # through-hole handle grasps with the complete Panda hand, not a visual-only
    # handle whose opening is too small for the wrist collision geometry.
    for name,p,size in [("handle_top",[.095,0,.091],[.063,.008,.006]),
                         ("handle_bottom",[.095,0,-.081],[.063,.008,.006]),
                         ("handle_bar",[.166,0,.005],[.008,.008,.092])]:
        add(BoxObject(name,size=size,rgba=color,density=350,friction=[2,.01,.0001]),p)
    return CompositeBodyObject("cube",parts,positions,object_quats=quats)


class ViewTask(Lift):
    def __init__(self, task="block", **kwargs):
        self.task_name = task
        super().__init__(**kwargs)

    def _load_model(self):
        ManipulationEnv._load_model(self)
        robot = self.robots[0].robot_model
        robot.set_base_xpos(robot.base_xpos_offset["table"](self.table_full_size[0]))
        arena = TableArena(table_full_size=self.table_full_size,table_friction=self.table_friction,table_offset=self.table_offset)
        arena.set_origin([0,0,0])
        camera_body = ET.SubElement(arena.worldbody,"body",name="active_camera",mocap="true",pos=".6 0 1.4")
        ET.SubElement(camera_body,"camera",name="activeview",fovy="70",pos="0 0 0")
        # Overview is recording only, never supplied to the agent.
        arena.set_camera(camera_name="overview",pos=[1.1,-1.1,1.8],quat=[.86,.34,.14,.34])
        self.cube = make_mug() if self.task_name=="mug" else BoxObject("cube",size=[.022]*3,
            rgba=[.05,.2,.9,1] if self.task_name=="occluded" else [.9,.04,.035,1],density=500,friction=[2,.01,.0001])
        self.scene_objects = [self.cube]
        if self.task_name=="occluded":
            self.scene_objects += [BoxObject("occluder",size=[.025,.065,.06],rgba=[.8,.08,.06,1],density=900),
                                   BoxObject("distractor",size=[.023]*3,rgba=[.8,.08,.06,1],density=500)]
            # Visual-only destination; its location must still be grounded from RGB by the agent.
            ET.SubElement(arena.worldbody,"geom",name="destination",type="box",pos="0 .20 .801",
                size=".07 .065 .001",rgba=".95 .8 .03 1",contype="0",conaffinity="0",group="1")
        self.model = ManipulationTask(arena,[r.robot_model for r in self.robots],self.scene_objects)
        visual=self.model.root.find("visual")
        if visual is None: visual=ET.SubElement(self.model.root,"visual")
        quality=visual.find("quality")
        if quality is None: quality=ET.SubElement(visual,"quality")
        # MSAA depth resolves at a sub-pixel sample on macOS; disable it so RGB-D
        # shares the explicitly calibrated pixel centers.
        quality.set("offsamples","0")

    def _reset_internal(self):
        ManipulationEnv._reset_internal(self)
        if not self.deterministic_reset:
            for i,obj in enumerate(self.scene_objects):
                if i==0:
                    xy = self.rng.uniform([-.06,-.07],[.05,.05])
                    yaw = self.rng.uniform(-np.pi,np.pi)
                elif i==1:
                    xy = np.array([.12,0])+self.rng.uniform(-.015,.015,2); yaw=0
                else:
                    xy = np.array([-.06,-.19])+self.rng.uniform(-.02,.02,2); yaw=self.rng.uniform(-np.pi,np.pi)
                pos = np.r_[xy,.805-obj.bottom_offset[2]]
                self.sim.data.set_joint_qpos(obj.joints[0],np.r_[pos,np.cos(yaw/2),0,0,np.sin(yaw/2)])

    def reward(self, action=None):
        return 0.0  # Task truth is reserved for Evaluator, not used as a controller signal.


class Simulator:
    def __init__(self, task, seed, size=512):
        config = load_composite_controller_config(robot="Panda")
        arm = config["body_parts"]["right"]
        grip = arm["gripper"]
        arm.clear(); arm.update(type="JOINT_POSITION",input_max=1,input_min=-1,
            output_max=.1,output_min=-.1,kp=200,damping_ratio=1,impedance_mode="fixed",
            kp_limits=[0,300],damping_ratio_limits=[0,10],qpos_limits=None,
            interpolation=None,ramp_ratio=.2,input_type="absolute",gripper=grip)
        self.env = ViewTask(task=task,robots="Panda",controller_configs=config,
            has_renderer=False,has_offscreen_renderer=True,use_camera_obs=False,
            use_object_obs=False,initialization_noise=None,control_freq=20,
            hard_reset=False,ignore_done=True,seed=seed)
        self.env.reset()
        # mj_saveLastXML depends on the last compiled XML globally. Snapshot
        # before the separate robot-only planning model is compiled.
        self.scene_xml=self.env.sim.model.get_xml()
        self.task = task; self.size = size; self.frame_id = 0
        self.robot = self.env.robots[0]
        self.joint_names = list(self.robot.robot_model.joints)
        self.qidx = list(self.robot._ref_joint_pos_indexes)
        self.gripidx = list(self.robot._ref_gripper_joint_pos_indexes["right"])
        self.view = (0.,80.,1.3)
        self.move_camera(*self.view)
        # Standalone robot XML is the only model exported to the planner.
        self.robot_xml = self.robot.robot_model.get_xml()
        self.eef_name = self.robot.gripper["right"].important_sites["grip_site"]
        self.states = []; self.actions = []; self.frames = []; self.phase = "observe"
        self.evaluator = None
        q = self.q
        for _ in range(30): self.step(q,-1,record=False)

    @property
    def q(self): return self.env.sim.data.qpos[self.qidx].copy()

    @property
    def gripper_q(self): return self.env.sim.data.qpos[self.gripidx].copy()

    def validate_initial_proxies(self,points):
        """Scene-builder validity check only; returns no object data to the planner.

        The initial robot must be clear even under the planner's conservative
        rigid-link representation. A 10 mm expansion covers sampling gaps.
        """
        import mujoco
        model=self.env.sim.model;data=self.env.sim.data
        for gid in range(model.ngeom):
            name=model.geom_id2name(gid) or ""
            if name.startswith(("robot0_","gripper0_","mount0_")):continue
            if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):continue
            p=(points-data.geom_xpos[gid])@data.geom_xmat[gid].reshape(3,3)
            size=model.geom_size[gid];typ=model.geom_type[gid]
            if typ==mujoco.mjtGeom.mjGEOM_BOX:
                hit=np.all(np.abs(p)<size+.01,axis=1)
            elif typ==mujoco.mjtGeom.mjGEOM_CYLINDER:
                hit=(np.linalg.norm(p[:,:2],axis=1)<size[0]+.01)&(np.abs(p[:,2])<size[1]+.01)
            elif typ==mujoco.mjtGeom.mjGEOM_PLANE:hit=p[:,2]<.01
            else:continue
            if np.any(hit):raise ValueError("invalid initial scene: robot proxy clearance")

    def move_camera(self,azimuth,elevation,distance):
        az = np.deg2rad(azimuth); el = np.deg2rad(elevation)
        focus = np.array([-.12,0,.95])  # Known workspace center, not a target pose.
        pos = focus+distance*np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
        z = focus-pos; z /= np.linalg.norm(z)
        x = np.cross(z,[0,0,1]); x/=np.linalg.norm(x)
        y = np.cross(z,x)
        R_gl = np.column_stack([x,-y,-z])
        quat = Rotation.from_matrix(R_gl).as_quat()[[3,0,1,2]]
        self.env.sim.data.set_mocap_pos("active_camera",pos)
        self.env.sim.data.set_mocap_quat("active_camera",quat)
        self.env.sim.forward(); self.view=(float(azimuth),float(elevation),float(distance))

    def observe(self):
        self.env.sim._render_context_offscreen.gl_ctx.make_current()
        rgb,d = self.env.sim.render(width=self.size,height=self.size,camera_name="activeview",depth=True)
        self.frame_id+=1
        K=get_camera_intrinsic_matrix(self.env.sim,"activeview",self.size,self.size)
        # Integer u,v index pixel CENTERS; the OpenGL viewport centers are i+0.5.
        K[:2,2]-=.5
        return Observation(self.frame_id,rgb[::-1].copy(),get_real_depth_map(self.env.sim,d[::-1]).copy(),
            K,
            get_camera_extrinsic_matrix(self.env.sim,"activeview"),self.q,self.gripper_q,
            float(self.env.sim.data.time),self.view)

    def step(self,q,gripper,record=True):
        # JOINT_POSITION expects physical radians when input_type=absolute.
        action = self.robot.create_action_vector({"right":np.asarray(q),"right_gripper":np.array([gripper])})
        self.env.step(action)
        if self.evaluator: self.evaluator.update(self.phase)
        if record:
            self.actions.append(action.copy())
            self.states.append(self.env.sim.get_state().flatten().copy())
            if len(self.actions)%2==0:
                self.env.sim._render_context_offscreen.gl_ctx.make_current()
                rgb=self.env.sim.render(width=640,height=480,camera_name="agentview")
                self.frames.append((rgb[::-1].copy(),self.phase,float(self.env.sim.data.time)))

    def close(self): self.env.close()
