from pathlib import Path
import json,time
import numpy as np
from PIL import Image
from view2grasp.sim import Simulator,INSTRUCTIONS
from view2grasp.planning import RobotModel,Planner
from view2grasp.mapping import DepthMap
from view2grasp.geometry import cloud,fit_table,grasp_candidates
from view2grasp.vlm import VLM

sim=Simulator("block",0)
out=Path("runs/diagnose"); out.mkdir(parents=True,exist_ok=True)
obs=sim.observe(); Image.fromarray(obs.rgb).save(out/"rgb.png")
robot=RobotModel(sim.robot_xml,sim.joint_names,sim.eef_name)
points,_=robot.points(obs.q,obs.gripper_q)
print("robot pose",robot.pose(),"sim pose",sim.env.sim.data.get_site_xpos(sim.eef_name),flush=True)
map_=DepthMap(); map_.certify_home(points); map_.integrate(obs,points)
map_.table_z=fit_table(cloud(obs),np.random.default_rng(0))
print("map",len(map_.surface),"table",map_.table_z,flush=True)
vlm=VLM(out/"vlm")
decision=vlm.decide(obs,INSTRUCTIONS["block"],[],True)
print(decision,flush=True)
gs,target=grasp_candidates(obs,decision,map_.table_z)
map_.set_target(target)
planner=Planner(robot,map_,np.random.default_rng(0))
print("candidates",len(gs),"target",target.min(0),target.max(0),"start valid",planner.valid(obs.q),planner.last_failure,flush=True)
for g in gs[:30:3]:
 q=robot.ik(g.position-g.approach*.08,g.rotation,obs.q)
 if q is not None:
  pts,pads=robot.points(q)
  state=map_.classify(pts)
  print("ik",g.ray_angle,"map counts",np.unique(state,return_counts=True),"bad bounds",pts[state!=1].min(0) if np.any(state!=1) else None,flush=True)
print("choose starting",flush=True)
t=time.monotonic(); result=planner.choose(obs.q,gs,max_candidates=24)
print("choose",time.monotonic()-t,result is not None,planner.last_failure,flush=True)
sim.close()
