from pathlib import Path
from PIL import Image
from view2grasp.sim import Simulator
from view2grasp.geometry import cloud, fit_table
import numpy as np

out=Path("runs/smoke"); out.mkdir(parents=True,exist_ok=True)
for task in ("block","mug","occluded"):
    sim=Simulator(task,0)
    obs=sim.observe()
    Image.fromarray(obs.rgb).save(out/f"{task}.png")
    print(task,"q",sim.q,"eef",sim.eef_name,"grip",sim.gripper_q,"depth",obs.depth.min(),obs.depth.max(),"table",fit_table(cloud(obs),np.random.default_rng(0)))
    (out/f"{task}_robot.xml").write_text(sim.robot_xml)
    sim.close()
