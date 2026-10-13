"""Video composition from real records; no synthesized robot actions."""
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio


def font(size):
    choices=[os.getenv("VIEW2GRASP_FONT",""),"/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc","/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for p in choices:
        if p and Path(p).exists(): return ImageFont.truetype(p,size)
    return ImageFont.load_default(size=size)


def replay_episode(episode,output):
    episode=Path(episode); record=json.loads((episode/"result.json").read_text())
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    data=np.load(episode/"trajectory.npz")
    if not len(data["states"]):
        with imageio.get_writer(output,fps=10,codec="libx264",quality=8,macro_block_size=1) as writer:
            for path in sorted((episode/"observations").glob("frame_*.png")):
                ground=path.with_name(path.name.replace("frame_","grounding_"))
                im=Image.new("RGB",(1280,720),(13,21,30))
                source=Image.open(ground if ground.exists() else path).convert("RGB").resize((640,640))
                im.paste(source,(320,40)); d=ImageDraw.Draw(im)
                d.text((25,10),f"{record['task']} | {record['strategy']} | seed {record['seed']} | {path.stem}",font=font(22),fill="white")
                d.text((25,686),"OBSERVATION-ONLY REPLAY / no robot motion executed / 2s per view, waiting condensed",font=font(19),fill="white")
                for _ in range(20): writer.append_data(np.asarray(im))
            im=Image.new("RGB",(1280,720),(13,21,30));d=ImageDraw.Draw(im)
            d.text((70,290),f"SUCCESS: {record['success']}  /  Grasp attempts: {record['attempts']}",font=font(35),fill="white")
            d.text((70,360),record['failure'],font=font(25),fill="white")
            for _ in range(40):writer.append_data(np.asarray(im))
        return
    import sys
    os.environ.setdefault('MUJOCO_GL','cgl' if sys.platform=='darwin' else 'egl')
    from robosuite.utils.binding_utils import MjSim,MjRenderContextOffscreen
    sim=MjSim.from_xml_string((episode/"scene.xml").read_text())
    try:
        camera='robot0_agentview_left' if record['task'] in ('NavigateKitchen','DeliverCanToBowl','DeliverBrewedCoffee') else 'overview'
        if camera not in sim.model.camera_names:
            raise ValueError('recording lacks the complete scene XML; historical snapshot is not state-replayable (original execution video is still available)')
        expected=len(sim.get_state().flatten())
        if data['states'].shape[1]!=expected:raise ValueError('recorded states do not match scene dimensions')
        context=MjRenderContextOffscreen(sim,device_id=-1,max_width=1280,max_height=720)
        # Match the live robosuite renderer: group 0 contains collision proxies,
        # while group 1 contains the visual meshes seen by the agent.
        context.vopt.geomgroup[0]=0
        context.vopt.geomgroup[1]=1
        sim.add_render_context(context)
        with imageio.get_writer(output,fps=20,codec="libx264",quality=8) as writer:
            for state in data["states"]:
                sim.set_state_from_flattened(state); sim.forward()
                im=sim.render(width=1280,height=720,camera_name=camera)[::-1]
                writer.append_data(im)
    finally:sim.free()


def compose_demo(episodes, output, annotations=None):
    from .demo import compose
    return compose(episodes, output, annotations)
