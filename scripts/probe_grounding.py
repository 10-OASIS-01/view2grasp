"""Real VLM capability probe over recorded RGB-D; NOT an online robot episode."""
import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image
from view2grasp.types import Observation
from view2grasp.vlm import VLM
from view2grasp.recording import save_observation


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('frame',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--instruction',default='Locate the graspable object and its graspable part.')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('probe output already exists')
    d=np.load(args.frame)
    obs=Observation(int(args.frame.stem.split('_')[-1]),np.array(Image.open(args.frame.with_suffix('.png'))),
        d['depth'],d['K'],d['T_wc'],d['q'],d['gripper_q'],float(d['timestamp']),tuple(d['view']))
    results=[]
    for kind,extra in [('points','Use contact points and object/part boxes.'),
                       ('box','Use object and grasp_part boxes, with no explicit grasp point.'),
                       ('polygon','Use polygon object and grasp_part regions; no contact point.'),
                       ('path','Describe a short image-space end_effector_path towards the target. It is only a 2D hint, not an executable trajectory. Use action abort to avoid claiming execution.')]:
        client=VLM(args.output/kind/'vlm')
        try:
            decision=client.decide(obs,args.instruction+' '+extra,[])
            save_observation(args.output/kind/'observations',obs,decision)
            results.append({'requested':kind,'valid_json':True,'returned':[g.kind for g in decision.groundings],
                            'action':decision.action,'calls':client.calls,'seconds':client.seconds})
        except Exception as exc:
            results.append({'requested':kind,'valid_json':False,'error':str(exc),'calls':client.calls})
        finally:client.client.close()
        print(json.dumps(results[-1]),flush=True)
    (args.output/'report.json').write_text(json.dumps({'kind':'live_vlm_on_recorded_images_not_online_episode',
        'source':str(args.frame.resolve()),'results':results},indent=2))

if __name__=='__main__':main()
