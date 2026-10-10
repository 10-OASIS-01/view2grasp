"""Trace stopped-lift obstacle samples to recorded RGB-D endpoints.

This diagnostic uses no evaluator/object truth and never mutates an episode.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from view2grasp.geometry import cloud,fit_table
from view2grasp.grounding import track_surface
from view2grasp.mapping import DepthMap
from view2grasp.planning import RobotModel
from view2grasp.sim import Simulator
from view2grasp.types import Observation


def load_observation(path):
    data=np.load(path)
    return Observation(int(path.stem.split('_')[-1]),np.asarray(Image.open(path.with_suffix('.png'))),
        data['depth'],data['K'],data['T_wc'],data['q'],data['gripper_q'],float(data['timestamp']),
        tuple(data['view']),str(data['camera_id']) if 'camera_id' in data else 'activeview')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('episode',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    manifest=json.loads((args.episode/'manifest.json').read_text())
    sim=Simulator(manifest['task'],manifest['seed'])
    robot=RobotModel(sim.robot_xml,sim.joint_names,sim.eef_name)
    try:
        depth_map=DepthMap(.005)
        depth_map.certify_home(robot.points(sim.q,sim.gripper_q)[0],robot.volumes(sim.q,sim.gripper_q))
        source_clouds=[];observations=[];robot_depths={};support_z=None
        for path in sorted((args.episode/'observations').glob('frame_*.npz')):
            obs=load_observation(path);robot_depth=robot.depth(obs)
            observations.append(obs);robot_depths[obs.frame_id]=robot_depth
            if support_z is None:support_z=fit_table(cloud(obs),np.random.default_rng(0))
            points=cloud(obs,2);measured=obs.depth[::2,::2].ravel()
            valid=np.isfinite(points).all(1)&np.isfinite(measured)&(measured>.05)&(measured<3.)
            valid &= ~(np.abs(robot_depth-obs.depth)<.0002)[::2,::2].ravel()
            source_clouds.append((obs.frame_id,points[valid]))
            depth_map.integrate(obs,robot.points(obs.q,obs.gripper_q)[0],robot_depth,
                robot.volumes(obs.q,obs.gripper_q))
        candidate_file=sorted(args.episode.glob('candidates_*.npz'))[0]
        target=np.load(candidate_file)['target'];target_tree=cKDTree(target)
        reference=target.copy();associated=[target]
        for obs in observations:
            seen=track_surface(obs,reference,robot_depths[obs.frame_id],support_z=support_z)
            if seen is None or not len(seen):continue
            associated.append(seen);reference=np.vstack([reference,seen])
            _,unique=np.unique(np.round(reference/depth_map.resolution).astype(np.int32),axis=0,return_index=True)
            reference=reference[unique]
        associated=np.vstack(associated);associated_tree=cKDTree(associated)
        lift=json.loads((args.episode/'lift_planning_diagnostic.json').read_text())
        invalid=np.asarray(lift['payload_invalid']['points'],dtype=float)
        traces=[]
        for point in invalid:
            distance,index=depth_map.tree.query(point);endpoint=depth_map.surface[index]
            sources=[]
            for frame,points in source_clouds:
                source_distance=cKDTree(points).query(endpoint)[0]
                if source_distance<=depth_map.resolution*1.8:
                    sources.append({'frame_id':frame,'distance_m':float(source_distance)})
            traces.append({'payload_query':point.tolist(),'nearest_endpoint':endpoint.tolist(),
                'endpoint_distance_m':float(distance),
                'endpoint_to_initial_target_m':float(target_tree.query(endpoint)[0]),
                'endpoint_to_tracked_target_m':float(associated_tree.query(endpoint)[0]),
                'query_to_initial_target_m':float(target_tree.query(point)[0]),
                'source_frames':sources})
        comparisons=[]
        for label,association in (('initial',target),('tracked_union',associated)):
            association_tree=cKDTree(association)
            for radius in (.002,depth_map.resolution*1.3):
                keep=association_tree.query(depth_map.surface,distance_upper_bound=radius)[0]>radius
                remaining=depth_map.surface[keep]
                tree=cKDTree(remaining) if len(remaining) else None
                distances=np.full(len(invalid),np.inf) if tree is None else tree.query(invalid)[0]
                comparisons.append({'association':label,'association_points':len(association),
                    'association_radius_m':radius,
                    'remaining_endpoints_within_occupied_radius':int(np.sum(distances<depth_map.resolution*1.3)),
                    'nearest_remaining_endpoint_m':distances.tolist()})
        report={'kind':'offline_observation_only_payload_endpoint_trace',
            'episode':str(args.episode),'candidate_file':candidate_file.name,
            'support_z':support_z,'tracked_association_points':len(associated),
            'occupied_radius_m':depth_map.resolution*1.3,'invalid_points':traces,
            'association_radius_comparison':comparisons}
        output=args.output or args.episode/'offline_payload_diagnostic.json'
        output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    finally:
        robot.close();sim.close()


if __name__=='__main__':main()
