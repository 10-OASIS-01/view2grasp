"""Offline sensor calibration audit. Privileged labels stay in this diagnostic."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from view2grasp.robocasa_backend import RoboCasaSimulator
from view2grasp.planning import RobotModel
from view2grasp.geometry import cloud
from view2grasp.mobile_cameras import CAMERAS


def main():
    out=Path('artifacts/mobile-depth-audit');out.mkdir(parents=True,exist_ok=True)
    sim=RoboCasaSimulator('NavigateKitchen',3,512,layout=2)
    robot=RobotModel(sim.robot_xml,sim.joint_names,sim.eef_name,include_base=True)
    try:
        native=sim.env.sim
        import mujoco
        ids=np.array([i for i in range(native.model.ngeom)
            if (native.model.geom_id2name(i) or '').startswith(('robot0_','gripper0_','mobilebase0_'))])
        positions=[]
        robot.set_auxiliary_state(sim.auxiliary_state())
        for i in range(robot.model.ngeom):
            name=robot.model.geom(i).name
            if name is None:continue
            try:other=native.model.geom_name2id(name)
            except ValueError:continue
            positions.append(float(np.linalg.norm(robot.data.geom_xpos[i]-native.data.geom_xpos[other])))
        volumes=robot.volumes(sim.q,sim.gripper_q)
        def inside_volumes(points):
            result=np.zeros(len(points),dtype=bool)
            for A,b,lo,hi in volumes:
                candidates=np.flatnonzero(~result&np.all((points>=lo)&(points<=hi),axis=1))
                if len(candidates):result[candidates]=np.all(points[candidates]@A.T+b<=1e-7,axis=1)
            return result
        cameras=[]
        for camera in CAMERAS:
            obs=sim.observe(camera);pred=robot.depth(obs)
            seg=native.render(width=512,height=512,camera_name=camera,segmentation=True)[::-1]
            truth=(seg[:,:,0]==int(mujoco.mjtObj.mjOBJ_GEOM))&np.isin(seg[:,:,1],ids)
            errors=np.abs(pred[truth]-obs.depth[truth])
            points=cloud(obs,2);measured=obs.depth[::2,::2].ravel()
            retained=np.isfinite(points).all(1)&np.isfinite(measured)&(measured>.05)&(measured<10)
            retained &= ~(np.abs(pred-obs.depth)<.0002)[::2,::2].ravel()
            retained_inside=retained&inside_volumes(points)
            truth_sampled=truth[::2,::2].ravel()
            cameras.append({'camera':camera,'truth_robot_pixels':int(truth.sum()),
                'depth_error_quantiles_m':np.quantile(errors,[0,.5,.9,.99,1]).tolist() if len(errors) else [],
                'matched_fraction':float(np.mean(errors<.0002)) if len(errors) else None,
                'retained_inside_robot_volumes':int(retained_inside.sum()),
                'retained_inside_native_robot_label':int(np.sum(retained_inside&truth_sampled)),
                'retained_inside_external_label':int(np.sum(retained_inside&~truth_sampled))})
            Image.fromarray(obs.rgb).save(out/f'{camera}.png')
            overlay=obs.rgb.copy();overlay[truth]=[255,0,0];overlay[truth&(np.abs(pred-obs.depth)<.0002)]=[0,255,0]
            Image.fromarray(overlay).save(out/f'{camera}_selfmask.png')
        report={'kind':'privileged_sensor_audit_not_agent_input',
            'geometry_position_max_m':max(positions),'cameras':cameras,
            'native_near_far':[float(native.model.vis.map.znear*native.model.stat.extent),float(native.model.vis.map.zfar*native.model.stat.extent)],
            'robot_near_far':[float(robot.model.vis.map.znear*robot.model.stat.extent),float(robot.model.vis.map.zfar*robot.model.stat.extent)]}
        (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    finally:robot.close();sim.close()


if __name__=='__main__':main()
