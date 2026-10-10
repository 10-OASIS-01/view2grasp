"""Locate the first unknown-space obstruction along an offline Cartesian path."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from view2grasp.sim import Simulator
from view2grasp.types import Observation,AgentDecision
from view2grasp.planning import RobotModel,Planner
from view2grasp.mapping import DepthMap
from view2grasp.geometry import cloud,fit_table,grasp_candidates,project

episode=Path('runs/dev4/block_active_0003');out=episode/'approach_diagnostic';out.mkdir(exist_ok=True)
sim=Simulator('block',3);robot=RobotModel(sim.robot_xml,sim.joint_names,sim.eef_name)
views=[]
for path in sorted((episode/'observations').glob('frame_*.npz')):
 d=np.load(path)
 views.append(Observation(int(path.stem.split('_')[-1]),np.asarray(Image.open(path.with_suffix('.png'))),
  d['depth'],d['K'],d['T_wc'],d['q'],d['gripper_q'],float(d['timestamp']),tuple(d['view'])))
first=views[0];obs=views[-1];m=DepthMap()
m.certify_home(robot.points(first.q,first.gripper_q)[0],robot.volumes(first.q,first.gripper_q))
for v in views:m.integrate(v,robot.points(v.q,v.gripper_q)[0],robot.depth(v),robot.volumes(v.q,v.gripper_q))
m.table_z=fit_table(cloud(first),np.random.default_rng(0))
d=AgentDecision.model_validate_json((episode/'observations/decision_005.json').read_text())
gs,target=grasp_candidates(obs,d,m.table_z,robot.pose(obs.q)[1]);m.set_target(target)
q=np.load(episode/'coverage_diagnostic/best.npz')['q'];p,R=robot.pose(q)
g=min(gs,key=lambda g:np.linalg.norm(g.position-g.approach*.08-p)+np.linalg.norm(Rotation.from_matrix(g.rotation@R.T).as_rotvec()))
planner=Planner(robot,m,np.random.default_rng(0));planner.gripper=obs.gripper_q
for off in np.linspace(.08,0,17):
 q=robot.ik(g.position-g.approach*off,g.rotation,q)
 if q is None:print('IK fail');break
 ok=planner.valid(q,off<.012);pts,pads=robot.points(q,obs.gripper_q);state=m.classify(pts);bad=state!=1
 print('offset',off,'valid',ok,'counts',np.unique(state,return_counts=True),flush=True)
 if not ok:
  start=0
  for gid,local,pad,name in robot.parts:
   n=len(local);ids=np.flatnonzero(bad[start:start+n])+start
   if len(ids):print('bad part',name,'n',len(ids),'points',pts[ids],flush=True)
   start+=n
  np.savez_compressed(out/'failure.npz',q=q,points=pts,state=state)
  for point in pts[bad]:
   distances=sorted(float(np.max(A@point+b)) for A,b,lo,hi in m.home_volumes)
   print('outside home volumes',distances[:5],flush=True)
  robot.set_q(obs.q,obs.gripper_q)
  for bid in range(robot.model.nbody):
   vertices=[]
   for gid in range(robot.model.ngeom):
    if robot.model.geom_bodyid[gid]!=bid or robot.model.geom_dataid[gid]<0:continue
    mid=robot.model.geom_dataid[gid]
    start=robot.model.mesh_vertadr[mid];n=robot.model.mesh_vertnum[mid]
    vertices.append(robot.model.mesh_vert[start:start+n]@robot.data.geom_xmat[gid].reshape(3,3).T+robot.data.geom_xpos[gid])
   if not vertices:continue
   hull=ConvexHull(np.vstack(vertices)).equations
   margin=np.max(pts[bad]@hull[:,:3].T+hull[:,3],axis=1)
   if np.min(margin)<.03:print('body hull',robot.model.body(bid).name,margin,flush=True)
  for v in views:
   im=Image.fromarray(v.rgb);draw=ImageDraw.Draw(im);uv,z=project(pts[bad],v.K,v.T_wc)
   for (u,vv),zz in zip(uv,z):
    if zz>0 and 0<=u<512 and 0<=vv<512:draw.ellipse((u-2,vv-2,u+2,vv+2),fill='red')
   im.save(out/f'bad_{v.frame_id}.png')
  for camera in [(0,80,1.3),(90,80,1.3),(-90,80,1.3),(180,80,1.3)]:
   sim.move_camera(*camera);v=sim.observe();m.integrate(v,robot.points(v.q,v.gripper_q)[0],robot.depth(v),robot.volumes(v.q,v.gripper_q))
   print('extra high view',camera,'valid',planner.valid(q,off<.012),'bad',planner.last_bad,flush=True)
  break
robot.close();sim.close()
