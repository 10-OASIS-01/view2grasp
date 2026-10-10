"""Offline diagnostic on recorded observations, not an online benchmark result."""
import sys,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from view2grasp.sim import Simulator
from view2grasp.types import Observation,AgentDecision
from view2grasp.planning import RobotModel,Planner
from view2grasp.mapping import DepthMap
from view2grasp.geometry import cloud,fit_table,grasp_candidates,project

episode=Path(sys.argv[1]);out=episode/"coverage_diagnostic";out.mkdir(exist_ok=True)
result=json.loads((episode/"result.json").read_text());sim=Simulator(result["task"],result["seed"])
robot=RobotModel(sim.robot_xml,sim.joint_names,sim.eef_name)
observations=[]
for path in sorted((episode/"observations").glob("frame_*.npz")):
 data=np.load(path)
 obs=Observation(int(path.stem.split('_')[-1]),np.asarray(Image.open(path.with_suffix('.png'))),
  data['depth'],data['K'],data['T_wc'],data['q'],data['gripper_q'],float(data['timestamp']),tuple(data['view']))
 observations.append(obs)
first=observations[0];m=DepthMap();m.certify_home(robot.points(first.q,first.gripper_q)[0],robot.volumes(first.q,first.gripper_q))
for obs in observations:m.integrate(obs,robot.points(obs.q,obs.gripper_q)[0],robot.depth(obs),robot.volumes(obs.q,obs.gripper_q))
m.table_z=fit_table(cloud(first),np.random.default_rng(0))
obs=observations[-1];decision=AgentDecision.model_validate_json((episode/'observations'/f'decision_{obs.frame_id:03d}.json').read_text())
_,R=robot.pose(obs.q);candidates,target=grasp_candidates(obs,decision,m.table_z,R);m.set_target(target)
candidates=sorted(candidates,key=lambda g:.08*g.score+np.linalg.norm(Rotation.from_matrix(g.rotation@R.T).as_rotvec()))
best=99999;rng=np.random.default_rng(0)
planner=Planner(robot,m,rng);planner.gripper=obs.gripper_q
for ci,g in enumerate(candidates[:12]):
 for si in range(8):
  seed=obs.q if si==0 else np.clip(obs.q+rng.normal(0,.25,7),robot.limits[:,0]+.02,robot.limits[:,1]-.02)
  q=robot.ik(g.position-g.approach*.08,g.rotation,seed)
  if q is None:continue
  pts,pads=robot.points(q,obs.gripper_q);state=m.classify(pts);bad=state!=1
  if sum(bad)<best:
   best=int(sum(bad));print('best',ci,si,'bad',best,'counts',np.unique(state,return_counts=True),flush=True)
   np.savez_compressed(out/'best.npz',q=q,points=pts,state=state)
   for view in observations:
    im=Image.fromarray(view.rgb);d=ImageDraw.Draw(im);uv,z=project(pts[bad],view.K,view.T_wc)
    for (u,v),depth in zip(uv,z):
     if depth>0 and 0<=u<im.width and 0<=v<im.height:d.ellipse((u-2,v-2,u+2,v+2),fill='red')
    im.save(out/f'unknown_{view.frame_id}.png')
   if best==0:print('valid',planner.valid(q),'approach',planner.approach(q,g) is not None,planner.last_failure,flush=True)
  if best==0:break
 if best==0:break
robot.close();sim.close()
