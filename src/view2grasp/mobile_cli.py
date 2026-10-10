"""Entrypoint executed only by the isolated RoboCasa interpreter."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['doctor','run','replay'])
    p.add_argument('--episode',type=Path)
    p.add_argument('--task',default='NavigateKitchen')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--strategy',choices=['fixed','random','active'],default='active')
    p.add_argument('--root',type=Path,default=Path('runs/mobile-dev'))
    p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--render',action='store_true')
    args=p.parse_args()
    from .robocasa_backend import installation_report,RoboCasaSimulator
    if args.command=='doctor':
        report=installation_report()
        if args.render:
            sim=None
            try:
                sim=RoboCasaSimulator(args.task,args.seed,256,layout=2)
                obs=sim.observe()
                from .geometry import cloud,project
                import numpy as np
                xyz=cloud(obs,8);uv,z=project(xyz,obs.K,obs.T_wc)
                if not np.isfinite(xyz).all() or not np.isfinite(uv).all():
                    raise ValueError('nonfinite RGB-D calibration output')
                v,u=np.mgrid[0:256:8,0:256:8]
                report.update(render_ok=True,reprojection_max_px=float(np.max(np.abs(uv-np.stack([u,v],-1).reshape(-1,2)))),
                              action_layout=sim.action_layout,base_pose=sim.base_pose.tolist())
                from .planning import RobotModel
                robot=RobotModel(sim.robot_xml,sim.joint_names,sim.eef_name,include_base=True)
                try:
                    robot.set_auxiliary_state(sim.auxiliary_state())
                    p,R=robot.pose(sim.q)
                    native=sim.env.sim.data.get_site_xpos(sim.eef_name)
                    points=robot.points(obs.q,obs.gripper_q)[0]
                    boundary=robot.boundary_points(obs.q,obs.gripper_q)
                    if not np.isfinite(points).all():raise ValueError('nonfinite robot collision proxy')
                    depth=robot.depth(obs)
                    matches=np.abs(depth-obs.depth)<.0002
                    visible=depth<10
                    report.update(robot_fk_error_m=float(np.linalg.norm(p-native)),
                        robot_collision_samples=len(points),robot_proxy_finite=True,
                        robot_boundary_samples=len(boundary),
                        self_depth_matched_pixels=int(matches.sum()),
                        self_depth_visible_pixels=int(visible.sum()),
                        self_depth_match_fraction=float(matches[visible].mean()) if visible.any() else None)
                finally:robot.close()
                if args.output:
                    from .recording import save_observation
                    save_observation(args.output.parent/'doctor-observation',obs)
            except Exception as exc:
                report.update(render_ok=False,render_error=f'{type(exc).__name__}: {exc}')
            finally:
                if sim:sim.close()
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        if args.render and not report.get('render_ok'):raise SystemExit(1)
    elif args.command=='replay':
        if args.episode is None or args.output is None:raise ValueError('replay requires episode and output')
        from .video import replay_episode
        replay_episode(args.episode,args.output)
    else:
        from .mobile_runner import run_mobile_episode
        from .config import load_config
        result=run_mobile_episode(args.task,args.seed,args.strategy,args.root,load_config(args.config))
        print(json.dumps(result,indent=2))

if __name__=='__main__':main()
