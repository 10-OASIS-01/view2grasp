"""Development-only next-view diagnostic; no online performance claim."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from view2grasp.sim import Simulator
from view2grasp.types import Observation,AgentDecision
from view2grasp.geometry import cloud,fit_table,grasp_candidates
from view2grasp.planning import RobotModel,Planner
from view2grasp.mapping import DepthMap
from view2grasp.view_advisor import swept_unknown,suggest_views

episode=Path('runs/dev4/block_active_0003');out=Path('runs/advisor_diagnostic');out.mkdir(exist_ok=True)
sim=Simulator('block',3);robot=RobotModel(sim.robot_xml,sim.joint_names,sim.eef_name)
first=sim.observe();data=np.load(episode/'observations/frame_005.npz')
obs=Observation(5,np.asarray(Image.open(episode/'observations/frame_005.png')),data['depth'],data['K'],data['T_wc'],data['q'],data['gripper_q'],float(data['timestamp']),tuple(data['view']))
m=DepthMap();m.certify_home(robot.points(obs.q,obs.gripper_q)[0],robot.volumes(obs.q,obs.gripper_q))
for v in [first,obs]:m.integrate(v,robot.points(v.q,v.gripper_q)[0],robot.depth(v))
m.table_z=fit_table(cloud(first),np.random.default_rng(0))
d=AgentDecision.model_validate_json((episode/'observations/decision_005.json').read_text())
R=robot.pose(obs.q)[1];gs,target=grasp_candidates(obs,d,m.table_z,R);m.set_target(target)
gs=sorted(gs,key=lambda g:.08*g.score+np.linalg.norm(Rotation.from_matrix(g.rotation@R.T).as_rotvec()))
points=swept_unknown(robot,m,obs,gs[0]);print('unknown samples',len(points),flush=True)
advice=suggest_views(robot,obs,points);print('advice',advice,flush=True)
(out/'advice.json').write_text(json.dumps(advice,indent=2))
for view in advice['views']:
 sim.move_camera(*view);v=sim.observe();Image.fromarray(v.rgb).save(out/f'view_{v.frame_id}.png')
 m.integrate(v,robot.points(v.q,v.gripper_q)[0],robot.depth(v))
planner=Planner(robot,m,np.random.default_rng(0));planner.gripper=obs.gripper_q
result=planner.choose(obs.q,gs,time_budget=120)
print('planned',result is not None,planner.last_failure,'checks',planner.checks,'closest',planner.closest_invalid,flush=True)
if result is not None:
 g,transit,approach=result
 np.savez_compressed(out/'plan.npz',transit=transit,approach=approach,position=g.position,rotation=g.rotation,target=target)
robot.close();sim.close()
