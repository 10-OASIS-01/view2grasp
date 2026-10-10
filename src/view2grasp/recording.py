from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio


def save_grounding_result(directory,obs,result):
    """Keep semantic proposals separate from geometrically filtered masks."""
    directory=Path(directory)
    stem=f'grounding_{obs.frame_id:03d}'
    (directory/f'{stem}_validation.json').write_text(json.dumps(result.report,indent=2))
    np.savez_compressed(directory/f'{stem}_masks.npz',object_mask=result.object_mask,part_mask=result.part_mask)
    rgb=obs.rgb.copy()
    rgb[result.object_mask]=(rgb[result.object_mask]*.6+np.array([0,220,80])*.4).astype(np.uint8)
    rgb[result.part_mask]=(rgb[result.part_mask]*.4+np.array([0,190,255])*.6).astype(np.uint8)
    im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
    draw.text((8,8),'Validated surface masks: object=green, grasp part=cyan',fill='white')
    im.save(directory/f'{stem}_geometry.png')


def save_observation(directory,obs,decision=None):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    Image.fromarray(obs.rgb).save(directory/f"frame_{obs.frame_id:03d}.png")
    np.savez_compressed(directory/f"frame_{obs.frame_id:03d}.npz",depth=obs.depth,K=obs.K,T_wc=obs.T_wc,
        q=obs.q,gripper_q=obs.gripper_q,timestamp=obs.timestamp,view=obs.view,
        camera_id=obs.camera_id,base_pose=np.array([]) if obs.base_pose is None else obs.base_pose,
        torso_q=np.array([]) if obs.torso_q is None else obs.torso_q)
    if decision:
        (directory/f"decision_{obs.frame_id:03d}.json").write_text(decision.model_dump_json(indent=2))
        im=Image.fromarray(obs.rgb); draw=ImageDraw.Draw(im); w,h=im.size
        if decision.point:
            x,y=np.array(decision.point)*[w-1,h-1]
            draw.ellipse((x-6,y-6,x+6,y+6),outline="yellow",width=3)
        if decision.bbox:
            b=np.array(decision.bbox)*[w-1,h-1,w-1,h-1]
            draw.rectangle(tuple(b),outline="lime",width=2)
        from .grounding import pixels,proposal_mask
        for proposal in decision.groundings:
            color={'object':'lime','grasp_part':'cyan','grasp_contact':'yellow',
                   'ground_path':'orange','place_region':'magenta'}.get(proposal.role,'white')
            pts=[tuple(p) for p in pixels(proposal.points,(h,w))]
            if proposal.kind in ('polygon','mask'):
                mask=proposal_mask(proposal,obs)
                overlay=Image.new('RGBA',im.size,(0,0,0,0))
                tint=Image.new('RGBA',im.size,color)
                alpha=Image.fromarray((mask*65).astype(np.uint8))
                overlay.paste(tint,(0,0),alpha)
                im=Image.alpha_composite(im.convert('RGBA'),overlay).convert('RGB')
                draw=ImageDraw.Draw(im)
            if proposal.box:
                b=np.array(proposal.box)*[w-1,h-1,w-1,h-1]
                draw.rectangle(tuple(b),outline=color,width=2)
            if proposal.kind=='path':
                draw.line(pts,fill=color,width=3)
            if proposal.kind in ('points','path'):
                for i,(x,y) in enumerate(pts):
                    draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
                    draw.text((x+5,y+5),str(i),fill=color)
            if pts:
                draw.text(pts[0],proposal.role,fill=color)
        im.save(directory/f"grounding_{obs.frame_id:03d}.png")


def save_episode(directory,sim,record,events):
    directory=Path(directory)
    (directory/"result.json").write_text(json.dumps(asdict(record),indent=2))
    (directory/"events.json").write_text(json.dumps(events,indent=2,ensure_ascii=False))
    # Privileged state recording is for offline playback only, not perception or planning.
    (directory/"scene.xml").write_text(sim.scene_xml)
    np.savez_compressed(directory/"trajectory.npz",states=np.asarray(sim.states),actions=np.asarray(sim.actions))
    # A complete visual audit also exists for episodes correctly stopped before motion.
    observations=sorted((directory/"observations").glob("frame_*.png"))
    if observations:
        with imageio.get_writer(directory/"episode.mp4",fps=10,codec="libx264",quality=7) as writer:
            for path in observations:
                idx=path.stem.split("_")[-1]
                grounding=path.with_name(f"grounding_{idx}.png")
                image=Image.open(grounding if grounding.exists() else path).convert("RGB").resize((640,640))
                im=Image.new("RGB",(960,720),(13,21,30)); im.paste(image,(160,48)); draw=ImageDraw.Draw(im)
                draw.text((20,12),f"{record.task} | {record.strategy} | seed {record.seed} | observation {idx}",fill="white")
                draw.text((20,697),"Observation audit: 2 seconds per view; inference waiting condensed. No interpolated robot motion.",fill="white")
                for _ in range(20): writer.append_data(np.asarray(im))
            for rgb,phase,stamp in sim.frames:
                im=Image.new("RGB",(960,720),(13,21,30)); im.paste(Image.fromarray(rgb).resize((960,720)))
                draw=ImageDraw.Draw(im); draw.text((20,12),f"Actual execution 1x | {phase} | {stamp:.2f}s",fill="white")
                writer.append_data(np.asarray(im))
            end=Image.new("RGB",(960,720),(13,21,30)); draw=ImageDraw.Draw(end)
            draw.text((40,270),f"Success: {record.success} | Executed attempts: {record.attempts}",fill="white")
            draw.text((40,300),record.failure or "Task verified by independent evaluator",fill="white")
            for _ in range(30): writer.append_data(np.asarray(end))
    if sim.frames:
        with imageio.get_writer(directory/"execution.mp4",fps=10,codec="libx264",quality=7,macro_block_size=16) as writer:
            for rgb,phase,stamp in sim.frames:
                im=Image.fromarray(rgb); draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,640,32),fill=(14,20,30))
                draw.text((12,9),f"{record.task} | seed {record.seed} | {record.strategy} | {phase} | sim {stamp:.2f}s",fill="white")
                writer.append_data(np.asarray(im))
