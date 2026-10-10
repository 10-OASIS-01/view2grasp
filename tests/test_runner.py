"""Offline integration smoke test. Mocked decisions never enter online metrics."""
from types import SimpleNamespace
import json
import numpy as np
from view2grasp.types import AgentDecision


def test_observation_grounding_planner_recording_pipeline(tmp_path,monkeypatch):
    from view2grasp import runner
    class RecordedDecision:
        def __init__(self,*args,**kwargs):
            self.calls=0;self.seconds=0.;self.client=SimpleNamespace(close=lambda:None)
        def decide(self,obs,*args,**kwargs):
            self.calls+=1
            rgb=obs.rgb.astype(int)
            v,u=np.where((rgb[:,:,0]>100)&(rgb[:,:,0]>rgb[:,:,1]*2)&(rgb[:,:,0]>rgb[:,:,2]*2))
            h,w=obs.depth.shape
            bbox=(float(u.min()/(w-1)),float(v.min()/(h-1)),float(u.max()/(w-1)),float(v.max()/(h-1)))
            return AgentDecision(action='grasp',frame_id=obs.frame_id,target='test red pixels',part='test surface',
                point=(float(np.median(u)/(w-1)),float(np.median(v)/(h-1))),bbox=bbox,part_bbox=bbox)
    calls=[]
    def reject(self,q,candidates,**kwargs):
        calls.append(len(candidates));self.last_failure='deliberate test rejection';return None
    monkeypatch.setattr(runner,'VLM',RecordedDecision)
    monkeypatch.setattr(runner.Planner,'choose',reject)
    monkeypatch.setattr(runner,'save_episode',lambda *args:None)
    result=runner.run_episode('block',3,'fixed',tmp_path,replay=tmp_path/'offline',
        config=runner.ExperimentConfig(max_observations=2))
    assert calls and calls[0]>0
    assert result['online'] is False and not result['success'] and result['attempts']==0
    assert result['failure']=='planning: deliberate test rejection'
    assert (tmp_path/'block_fixed_0003/source_snapshot.zip').exists()
