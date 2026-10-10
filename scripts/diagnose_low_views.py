"""Offline coverage experiment, never an online evaluation episode."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from view2grasp.sim import Simulator
from view2grasp.types import Observation, AgentDecision
from view2grasp.planning import RobotModel, Planner
from view2grasp.mapping import DepthMap
from view2grasp.geometry import cloud, fit_table, grasp_candidates,project

episode=Path('runs/dev4/block_active_0003')
out=Path('runs/low_view_diagnostic'); out.mkdir(exist_ok=True)
sim=Simulator('block',3); robot=RobotModel(sim.robot_xml,sim.joint_names,sim.eef_name)
data=np.load(episode/'observations/frame_005.npz')
obs=Observation(5,np.asarray(Image.open(episode/'observations/frame_005.png')),
 data['depth'],data['K'],data['T_wc'],data['q'],data['gripper_q'],float(data['timestamp']),tuple(data['view']))
m=DepthMap();m.certify_home(robot.points(obs.q,obs.gripper_q)[0],robot.volumes(obs.q,obs.gripper_q))
sim.move_camera(0,80,1.3);v=sim.observe();m.integrate(v,robot.points(v.q,v.gripper_q)[0],robot.depth(v),robot.volumes(v.q,v.gripper_q))
for az,el,dist in [(120,0,1.3),(-120,0,1.3),(0,0,1.3)]:
 sim.move_camera(az,el,dist); view=sim.observe()
 Image.fromarray(view.rgb).save(out/f'view_{az}.png')
 m.integrate(view,robot.points(view.q,view.gripper_q)[0],robot.depth(view),robot.volumes(view.q,view.gripper_q))
m.integrate(obs,robot.points(obs.q,obs.gripper_q)[0],robot.depth(obs),robot.volumes(obs.q,obs.gripper_q))
m.table_z=fit_table(cloud(obs),np.random.default_rng(0))
decision=AgentDecision.model_validate_json((episode/'observations/decision_005.json').read_text())
candidates,target=grasp_candidates(obs,decision,m.table_z,robot.pose(obs.q)[1]);m.set_target(target)
planner=Planner(robot,m,np.random.default_rng(0));planner.gripper=obs.gripper_q
print('start',planner.valid(obs.q),planner.last_failure,flush=True)
robot.set_q(sim.q,sim.gripper_q)
diff=[]
for i in range(robot.model.ngeom):
 name=robot.model.geom(i).name
 if name is None:continue
 try:j=sim.env.sim.model.geom_name2id(name)
 except ValueError:continue
 diff.append((float(np.linalg.norm(robot.data.geom_xpos[i]-sim.env.sim.data.geom_xpos[j])),name))
print('geom mismatch',sorted(diff,reverse=True)[:3],flush=True)
pts,_=robot.points(obs.q,obs.gripper_q);state=m.classify(pts)
print('start bad',pts[state!=1],'bad endpoints',m.surface[m.tree.query(pts[state!=1])[1]],flush=True)
for view in m.views:
 endpoints=m.surface[m.tree.query(pts[state!=1])[1]]
 uv,z=project(endpoints,view.K,view.T_wc);uv=np.rint(uv).astype(int)
 for (u,v),zz in zip(uv,z):
  if 0<=u<512 and 0<=v<512:
   own=robot.depth(view)
   print('endpoint view',view.view,'uv',u,v,'z',zz,'scene',view.depth[v,u],'self',own[v,u],'qdelta',np.max(abs(view.q-obs.q)),flush=True)
   robot.renderer.disable_depth_rendering();Image.fromarray(robot.renderer.render()).save(out/f'self_{view.view[0]}_{view.view[1]}.png');robot.renderer.enable_depth_rendering()
if '--coverage-only' in sys.argv:
 robot.close();sim.close();sys.exit(0)
if '--transit-only' in sys.argv:
 R=robot.pose(obs.q)[1]
 candidates=sorted(candidates,key=lambda g:.08*g.score+np.linalg.norm(Rotation.from_matrix(g.rotation@R.T).as_rotvec()))
 for index,g in enumerate(candidates[:8]):
  q=robot.ik(g.position-g.approach*.08,g.rotation,obs.q)
  if q is None or not planner.valid(q):continue
  path=planner.plan(obs.q,q,timeout=5)
  print('transit',index,path is not None,planner.last_failure,'checks',planner.checks,'trees',getattr(planner,'last_tree_sizes',None),'closest',planner.closest_invalid,flush=True)
  if path is not None:
   np.savez_compressed(out/'transit.npz',path=path,position=g.position,rotation=g.rotation,approach_axis=g.approach,target=target)
   break
 robot.close();sim.close();sys.exit(0)
result=planner.choose(obs.q,candidates,time_budget=120)
print('chosen',result is not None,planner.last_failure,'checks',planner.checks,flush=True)
print('failure counts',planner.failure_counts,'best bad',planner.best_bad,flush=True)
print('closest invalid',planner.closest_invalid,flush=True)
if result is not None:
 g,transit,approach=result
 np.savez_compressed(out/'plan.npz',position=g.position,rotation=g.rotation,transit=transit,approach=approach,target=target)
robot.close();sim.close()
