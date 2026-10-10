"""Verify real recorded VLM outputs against depth, optionally through SAM."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from view2grasp.types import Observation,AgentDecision
from view2grasp.grounding import GroundingResolver,SAMSegmenter
from view2grasp.geometry import cloud,fit_table


def main():
    p=argparse.ArgumentParser();p.add_argument('probe',type=Path);p.add_argument('--sam',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    resolver=GroundingResolver(SAMSegmenter(args.sam) if args.sam else None)
    rows=[]
    for folder in sorted(args.probe.iterdir()):
        if not folder.is_dir():continue
        files=list((folder/'observations').glob('frame_*.npz'))
        if not files:continue
        f=files[0];d=np.load(f);idx=int(f.stem.split('_')[-1])
        obs=Observation(idx,np.array(Image.open(f.with_suffix('.png'))),d['depth'],d['K'],d['T_wc'],
            d['q'],d['gripper_q'],float(d['timestamp']),tuple(d['view']))
        decision=AgentDecision.model_validate_json((folder/'observations'/f'decision_{idx:03d}.json').read_text())
        if decision.action!='grasp':continue
        try:
            result=resolver.resolve_grasp(obs,decision,fit_table(cloud(obs),np.random.default_rng(0)))
            row={'representation':folder.name,**result.report}
            np.savez_compressed(args.output/f'{folder.name}.npz',object_mask=result.object_mask,part_mask=result.part_mask)
            rgb=obs.rgb.copy();rgb[result.part_mask]=(rgb[result.part_mask]*.4+np.array([0,180,220])*.6).astype(np.uint8)
            Image.fromarray(rgb).save(args.output/f'{folder.name}.png')
        except Exception as exc:row={'representation':folder.name,'validated':False,'error':str(exc)}
        rows.append(row);print(json.dumps(row),flush=True)
    (args.output/'report.json').write_text(json.dumps({'kind':'offline_geometry_validation','sam_calls':resolver.calls,
        'seconds':resolver.seconds,'results':rows},indent=2))

if __name__=='__main__':main()
