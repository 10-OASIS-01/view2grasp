"""Reconstruct clearance solely from recorded RGB-D and robot proprioception."""
import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from view2grasp.types import Observation
from view2grasp.robocasa_backend import RoboCasaSimulator
from view2grasp.planning import RobotModel
from view2grasp.mapping import DepthMap
from view2grasp.geometry import cloud
from view2grasp.geometry import project
from view2grasp.mobile_runner import support_plane
from view2grasp.navigation import NavigationCollision,transform_body


def main():
    parser=argparse.ArgumentParser();parser.add_argument('episode',type=Path)
    parser.add_argument('--current-rig',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    metadata=json.loads((args.episode/'manifest.json').read_text())
    first_file=next(iter(sorted((args.episode/'observations').glob('frame_*.npz'))))
    first_data=np.load(first_file);size=int(first_data['depth'].shape[0])
    sim=RoboCasaSimulator(metadata['task'],metadata['seed'],size,layout=2)
    robot=RobotModel(sim.robot_xml,sim.joint_names,sim.eef_name,include_base=True)
    try:
        robot.set_auxiliary_state(sim.auxiliary_state())
        m=DepthMap(.005,workspace_bounds=None,max_depth=10)
        m.certify_home(robot.points(sim.q,sim.gripper_q)[0],robot.volumes(sim.q,sim.gripper_q))
        seen=set();trajectory=np.load(args.episode/'trajectory.npz')['states']
        times=trajectory[:,0]
        if args.current_rig:
            from view2grasp.robocasa_backend import CAMERAS
            observations=[sim.observe(camera) for camera in CAMERAS]
        else:
            observations=[]
            files=list((args.episode/'observations').glob('frame_*.npz'))+list(
                (args.episode/'safety_observations').glob('frame_*.npz'))
            files=sorted(files,key=lambda f:int(f.stem.split('_')[-1]))
            for f in files:
                d=np.load(f);camera=str(d['camera_id'])
                observations.append(Observation(int(f.stem.split('_')[-1]),np.array(Image.open(f.with_suffix('.png'))),
                    d['depth'],d['K'],d['T_wc'],d['q'],d['gripper_q'],float(d['timestamp']),tuple(d['view']),camera))
        source_clouds=[]
        for obs in observations:
            seen.add(obs.camera_id)
            if not args.current_rig:
                index=int(np.argmin(np.abs(times-obs.timestamp)))
                # The first observation precedes the first recorded control
                # state by one 20 Hz step; later observations align exactly.
                if abs(times[index]-obs.timestamp)>.051:raise ValueError('cannot align observation to simulator state')
                sim.env.sim.set_state_from_flattened(trajectory[index]);sim.env.sim.forward()
                robot.set_auxiliary_state(sim.auxiliary_state())
            robot_depth=robot.depth(obs)
            points=cloud(obs,2)
            measured=obs.depth[::2,::2].ravel()
            valid=np.isfinite(points).all(axis=1)&np.isfinite(measured)&(measured>.05)&(measured<m.max_depth)
            valid &= ~(np.abs(robot_depth-obs.depth)<.0002)[::2,::2].ravel()
            source_points=points[valid]
            source_clouds.append((f'{obs.frame_id}:{obs.camera_id}',
                cKDTree(source_points) if len(source_points) else None))
            m.integrate(obs,robot.points(obs.q,obs.gripper_q)[0],robot_depth,
                robot.volumes(obs.q,obs.gripper_q),robot.visual_volumes(obs.q,obs.gripper_q))
            m.certify_robot_state(robot.points(obs.q,obs.gripper_q)[0],robot.volumes(obs.q,obs.gripper_q))
        if not args.current_rig:
            sim.env.sim.set_state_from_flattened(trajectory[-1]);sim.env.sim.forward()
            robot.set_auxiliary_state(sim.auxiliary_state())
        floor_z,floor=support_plane(m.surface)
        world=robot.boundary_points(sim.q,sim.gripper_q)
        _,indices=np.unique(np.round(world/.02).astype(int),axis=0,return_index=True);world=world[indices]
        base=sim.base_pose
        body=transform_body(world-np.r_[base[:2],0],[0,0,-base[2]])
        floor=np.vstack([floor,world[world[:,2]<=floor_z+.025]])
        checker=NavigationCollision(m,body,floor,floor_z,shadow_pose=base,shadow_distance=.03)
        volumes=robot.volumes(sim.q,sim.gripper_q)
        def endpoint_inside_robot(point):
            for A,b,lo,hi in volumes:
                if np.all(point>=lo)&np.all(point<=hi)&np.all(point@A.T+b<=1e-7):return True
            return False
        rows=[]
        for delta in ((0,0,0),(.02,0,0),(-.02,0,0),(0,.02,0),(0,-.02,0),(0,0,.025),(0,0,-.025)):
            pose=base+delta;valid=checker(pose);points=transform_body(body,pose);low=points[:,2]<=floor_z+.025
            state=m.classify(points[~low]);bad=points[~low][state!=1]
            coverage=[]
            for obs,robot_mask in zip(m.views,m.self_masks):
                uv,z=project(bad,obs.K,obs.T_wc)
                uv=np.rint(np.nan_to_num(uv,nan=-1,posinf=-1,neginf=-1)).astype(int)
                h,w=obs.depth.shape
                in_fov=(z>.01)&(uv[:,0]>=1)&(uv[:,0]<w-1)&(uv[:,1]>=1)&(uv[:,1]<h-1)
                ids=np.flatnonzero(in_fov);measured=np.zeros(len(bad),dtype=bool);free=np.zeros(len(bad),dtype=bool)
                if len(ids):
                    u,v=uv[ids].T;d=obs.depth[v,u]
                    measured[ids]=np.isfinite(d)&(d>.05)
                    tolerance=np.where(robot_mask[v,u],.002,-.0002)
                    free[ids]=measured[ids]&(z[ids]<np.minimum(d,m.max_depth)+tolerance)
                coverage.append({'camera':obs.camera_id,'in_fov':int(in_fov.sum()),
                    'measured':int(measured.sum()),'free':int(free.sum()),
                    'occluded_or_behind_return':int(np.sum(measured&~free))})
            traces=[]
            for point in bad[:20]:
                distance,index=m.tree.query(point)
                endpoint=m.surface[index]
                source=[]
                for camera,source_tree in source_clouds:
                    if source_tree is not None:
                        source_distance=source_tree.query(endpoint)[0]
                        if source_distance<=m.resolution*1.8:source.append({'camera':camera,'distance_m':float(source_distance)})
                traces.append({'query_point':point.tolist(),'nearest_endpoint':endpoint.tolist(),
                    'distance_m':float(distance),'endpoint_inside_current_robot':endpoint_inside_robot(endpoint),
                    'source_candidates':source})
            rows.append({'world_delta':delta,'valid':valid,'reason':checker.last_failure,
                'unknown':int(np.sum(state==0)),'occupied':int(np.sum(state==2)),
                'floor_missing':int(np.sum(checker.floor.query(points[low,:2])[0]>.035)),
                'first_invalid_points':bad[:8].tolist(),'coverage':coverage,'invalid_traces':traces})
        report={'kind':'offline_observation_only_clearance','query':'complete collision-body boundary sweep',
            'boundary_samples':len(world),'filled_samples':len(robot.points(sim.q,sim.gripper_q)[0]),
            'observation_count':len(observations),'cameras':sorted(seen),'floor_z':floor_z,
            'final_base_pose':base.tolist(),'states':rows}
        output=args.output or args.episode/'clearance_diagnostic.json'
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    finally:robot.close();sim.close()


if __name__=='__main__':main()
