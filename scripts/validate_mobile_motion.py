"""Targeted RoboCasa controller smoke; no VLM and no task success claim."""
from pathlib import Path
import json
import numpy as np
from view2grasp.robocasa_backend import RoboCasaSimulator


def main():
    sim=RoboCasaSimulator('NavigateKitchen',3,128,layout=2)
    try:
        initial_q=sim.initial_arm.copy();hold_q=sim.q.copy();start=sim.base_pose.copy()
        for _ in range(4):sim.step(hold_q,-1,np.array([.05,0,0]),record=False)
        for _ in range(3):sim.step(hold_q,-1,np.zeros(3),record=False)
        end=sim.base_pose.copy()
        report={'kind':'robocasa_controller_smoke_not_task_result',
            'initial_arm_rad':initial_q.tolist(),'start_base_pose':start.tolist(),
            'end_base_pose':end.tolist(),'translation_m':float(np.linalg.norm(end[:2]-start[:2])),
            'arm_hold_error_rad':float(np.max(np.abs(sim.q-hold_q))),
            'reset_pose_error_rad':float(np.max(np.abs(hold_q-initial_q))),
            'steps_driving':4,'steps_settling':3}
        output=Path(__file__).resolve().parents[1]/'artifacts/robocasa-mobile-motion-smoke.json'
        output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
        assert report['translation_m']>.002
        assert report['arm_hold_error_rad']<.02 and report['reset_pose_error_rad']<1e-5
    finally:sim.close()


if __name__=='__main__':main()
