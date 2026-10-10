"""Privileged scoring. Never passed to agent, geometry, mapping or planner."""
import numpy as np


class Evaluator:
    def __init__(self,sim):
        self.sim=sim; env=sim.env
        self.body=env.sim.model.body_name2id(env.cube.root_body)
        self.initial_z=float(env.sim.data.body_xpos[self.body,2])
        self.initial_others={env.sim.model.body_name2id(o.root_body):env.sim.data.get_body_xpos(o.root_body).copy()
                             for o in env.scene_objects[1:]}
        self.target=set(env.sim.model.geom_name2id(n) for n in env.cube.contact_geoms)
        self.handle={i for i in self.target if "handle" in (env.sim.model.geom_id2name(i) or "")}
        grip=sim.robot.gripper["right"]
        self.left=set(env.sim.model.geom_name2id(n) for n in grip.important_geoms["left_finger"])
        self.right=set(env.sim.model.geom_name2id(n) for n in grip.important_geoms["right_finger"])
        self.robot_geoms={i for i in range(env.sim.model.ngeom) if (env.sim.model.geom_id2name(i) or "").startswith(("robot0_","gripper0_"))}
        self.count=0; self.lift_count=0;self.first_grasp_success=False;self.success=False; self.collision=False; self.history=[]

    def update(self,phase):
        sim=self.sim.env.sim
        touching_l=set(); touching_r=set(); unintended=False
        for c in sim.data.contact:
            a,b=int(c.geom1),int(c.geom2)
            if a in self.target and b in self.left: touching_l.add(a)
            if b in self.target and a in self.left: touching_l.add(b)
            if a in self.target and b in self.right: touching_r.add(a)
            if b in self.target and a in self.right: touching_r.add(b)
            robot_a=a in self.robot_geoms; robot_b=b in self.robot_geoms
            if robot_a!=robot_b and c.dist<-.001:
                robot=a if robot_a else b; other=b if robot_a else a
                expected=robot in (self.left|self.right) and other in self.target and phase in ("approach","close","lift","hold","transfer","lower","release")
                if not expected: unintended=True
        self.collision |= unintended
        p=sim.data.body_xpos[self.body].copy()
        grasp=bool(touching_l and touching_r)
        if self.sim.task=="mug":
            grasp=bool(touching_l & self.handle and touching_r & self.handle) and not ((touching_l|touching_r)-self.handle)
        lifted=p[2]-self.initial_z>=.08 and grasp
        self.lift_count=self.lift_count+1 if lifted else 0
        if getattr(self.sim,"episode_attempt",0)==1 and self.lift_count>=40:self.first_grasp_success=True
        stable_others=all(np.linalg.norm(sim.data.body_xpos[b]-p0)<.025 and sim.data.body_xmat[b].reshape(3,3)[2,2]>.9 for b,p0 in self.initial_others.items())
        if self.sim.task=="occluded":
            velocity=np.linalg.norm(sim.data.cvel[self.body])
            condition=abs(p[0])<.048 and abs(p[1]-.20)<.043 and abs(p[2]-self.initial_z)<.015 and not grasp and velocity<.03 and stable_others
        else: condition=lifted
        self.count=self.count+1 if condition else 0
        self.success |= self.count>=40
        self.history.append({"time":float(sim.data.time),"phase":phase,"position":p.tolist(),
                             "grasp":grasp,"condition":bool(condition),"stable_steps":self.count,"collision":unintended})

    def result(self):
        return {"success":bool(self.success),"collision_free_success":bool(self.success and not self.collision),
            "first_grasp_success":bool(self.first_grasp_success),"collision":bool(self.collision),"history":self.history}


def wilson(successes,n):
    if not n: return [0.,1.]
    z=1.96; p=successes/n; denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [float(center-half),float(center+half)]


def failure_category(reason):
    reason=reason.lower()
    if "vlm" in reason or "service" in reason:return "service"
    if "grounding" in reason or "json" in reason or "depth edge" in reason:return "grounding"
    if "unknown space" in reason or "map" in reason:return "mapping"
    if "width" in reason or "contact geometry" in reason:return "geometry"
    if "planning" in reason or "ik" in reason or "rrt" in reason:return "planning"
    if "execution" in reason or "tracking" in reason:return "execution"
    if "abort" in reason:return "agent_abort"
    if "interrupt" in reason:return "interrupted"
    return "other"
