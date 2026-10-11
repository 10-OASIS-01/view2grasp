"""Privileged mobile scoring and scene eligibility; never used by planners."""
import numpy as np


def scene_eligibility(sim):
    if sim.task == 'NavigateKitchen':return {'eligible':True}
    env=sim.env
    source_fixture='coffee_machine' if sim.task=='DeliverBrewedCoffee' else 'source_counter'
    target_object='breakfast_plate' if sim.task=='DeliverBrewedCoffee' else 'target_bowl'
    source=np.asarray(getattr(env,source_fixture).pos)[:2]
    destination=np.asarray(env.dining_counter.pos)[:2]
    separation=float(np.linalg.norm(source-destination))
    # A triangle-inequality reach bound deliberately overestimates the Panda
    # workspace. Ignoring joint limits cannot falsely certify infeasibility.
    model=env.sim.model
    joint=model.joint_name2id(sim.joint_names[0]);shoulder=int(model.jnt_bodyid[joint])
    site=model.site_name2id(sim.eef_name);body=int(model.site_bodyid[site])
    reach=float(np.linalg.norm(model.site_pos[site]))
    while body!=shoulder and body>0:
        reach+=float(np.linalg.norm(model.body_pos[body]));body=int(model.body_parentid[body])
    if body!=shoulder:raise ValueError('cannot establish fixed-base arm reach bound')
    # Include gripper contact offset and a generous object-center offset. The
    # full 30 cm official plate-neighborhood is subtracted from goal distance.
    reach+=.30
    plate=env.sim.data.body_xpos[env.obj_body_id[target_object]][:2]
    shoulder_xy=env.sim.data.body_xpos[shoulder][:2]
    closest_goal=float(np.linalg.norm(plate-shoulder_xy)-.30)
    unreachable=closest_goal>reach
    # This scene-generation filter does not export coordinates to the policy.
    return {'eligible':bool(separation>=1. and unreachable),'source_destination_separation_m':separation,
            'fixed_base_reach_upper_bound_m':reach,'closest_valid_destination_distance_m':closest_goal,
            'fixed_base_placement_provably_unreachable':bool(unreachable),
            'rule':'fixture centers >=1 m AND entire official 30 cm destination neighborhood outside conservative fixed-base arm reach (+30 cm grasp/object allowance)'}


class MobileEvaluator:
    def __init__(self,sim):
        self.sim=sim;self.history=[];self.collision=False;self.stable=0;self.native=False
        self.initial_base=sim.base_pose.copy();self.last_base=self.initial_base.copy()
        self.path_length=0.;self.held_path_length=0.
        self.first_grasp_success=None;self.lift_started=None;self.last_attempt=0
        self.target_body=None;self.initial_target_z=None
        env=sim.env
        self.robot_geoms={i for i in range(env.sim.model.ngeom)
            if (env.sim.model.geom_id2name(i) or '').startswith(('robot0_','gripper0_','mobilebase0_'))}
        grip=sim.robot.gripper['right']
        self.fingers={env.sim.model.geom_name2id(n) for key in ('left_finger','right_finger') for n in grip.important_geoms[key]}
        self.finger_sides=[{env.sim.model.geom_name2id(n) for n in grip.important_geoms[key]}
                           for key in ('left_finger','right_finger')]
        self.target=set()
        target_name={'DeliverBrewedCoffee':'brewed_coffee','DeliverCanToBowl':'soda_can'}.get(sim.task)
        if target_name:
            self.target={env.sim.model.geom_name2id(n) for n in env.objects[target_name].contact_geoms}
            self.target_body=env.obj_body_id[target_name]
            self.initial_target_z=float(env.sim.data.body_xpos[self.target_body,2])

    def update(self):
        env=self.sim.env
        base=self.sim.base_pose
        distance=float(np.linalg.norm(base[:2]-self.last_base[:2]));self.path_length+=distance
        contact_sides=set()
        for c in env.sim.data.contact:
            a,b=int(c.geom1),int(c.geom2)
            ra,rb=a in self.robot_geoms,b in self.robot_geoms
            if ra==rb:continue
            robot,other=(a,b) if ra else (b,a)
            if robot in self.fingers and other in self.target:
                if c.dist<=0:contact_sides.update(i for i,side in enumerate(self.finger_sides) if robot in side)
                continue
            name=env.sim.model.geom_id2name(other) or ''
            # Expected base-floor support, never arm-wall/counter contact.
            rn=env.sim.model.geom_id2name(robot) or ''
            if 'floor' in name.lower() and rn.startswith('mobilebase0_'):continue
            if c.dist<-.001:self.collision=True
        contacts=len(contact_sides)==2
        if contacts:self.held_path_length+=distance
        attempt=getattr(self.sim,'episode_attempt',0)
        now=float(env.sim.data.time)
        lifted=False;lift_seconds=0.
        if self.target_body is not None:
            lifted=bool(contacts and env.sim.data.body_xpos[self.target_body,2]-self.initial_target_z>=.08)
            if attempt>0 and self.first_grasp_success is None:self.first_grasp_success=False
            if attempt!=self.last_attempt or not lifted:self.lift_started=None
            if lifted and attempt>0:
                if self.lift_started is None:self.lift_started=now
                lift_seconds=now-self.lift_started
                if attempt==1 and lift_seconds>=2.-1e-9:self.first_grasp_success=True
        self.last_attempt=attempt
        self.last_base=base.copy()
        self.native=bool(env._check_success())
        self.stable=self.stable+1 if self.native else 0
        self.history.append({'time':now,'native':self.native,'phase':self.sim.phase,
                             'stable_steps':self.stable,'collision':self.collision,'held_contact':contacts,
                             'attempt':attempt,'lifted':lifted,'stable_lift_seconds':lift_seconds})

    def result(self):
        stable=self.native if self.sim.task=='NavigateKitchen' else self.stable>=40
        mobile=(self.held_path_length>=.5) if self.sim.task in ('DeliverBrewedCoffee','DeliverCanToBowl') else True
        return {'success':bool(stable and mobile),'collision_free_success':bool(stable and mobile and not self.collision),'official_success':self.native,
            'stable_success':bool(stable),'collision':self.collision,'path_length_m':self.path_length,
            'first_grasp_success':self.first_grasp_success,
            'held_path_length_m':self.held_path_length,'required_held_transport_met':mobile,'history':self.history}
