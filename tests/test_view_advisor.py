from types import SimpleNamespace
import numpy as np
from test_geometry import observation
from dataclasses import replace
from view2grasp.view_advisor import suggest_views,suggest_mounted_cameras


def test_advisor_never_mutates_observations_or_acquires_scene_images():
    obs=observation();before=obs.T_wc.copy();calls=[]
    def depth(hypothetical):
        calls.append(hypothetical.view)
        return np.full_like(hypothetical.depth,10.)
    # Its sole rendering dependency is the caller's independent robot model.
    result=suggest_views(SimpleNamespace(depth=depth),obs,np.array([[0.,0.,.95]]),budget=1,beam_width=2)
    assert len(calls)==60
    assert len(result['views'])==1
    assert np.array_equal(obs.T_wc,before)
    assert 'NOT acquired' in result['note']


def test_required_robot_coverage_precedes_payload_coverage(monkeypatch):
    import view2grasp.view_advisor as advisor
    obs=observation();payload=np.zeros((10,3));required=np.ones((2,3))
    def visible(robot,obs,points,view,observed_points=None):
        if points is required:
            return np.array([True,True]) if view[0]==-180 else np.array([True,False])
        return np.r_[True,np.zeros(9,dtype=bool)] if view[0]==-180 else np.ones(10,dtype=bool)
    monkeypatch.setattr(advisor,'_visible',visible)
    result=advisor.suggest_views(None,obs,payload,budget=1,required_points=required)
    assert result['views'][0][0]==-180
    assert result['predicted_required_coverage']==1.


def test_mounted_advisor_uses_calibration_without_scene_acquisition():
    obs=observation();calls=[]
    observations=[replace(obs,camera_id=name) for name in ('left','right')]
    def depth(calibrated):
        calls.append(calibrated.camera_id)
        return np.full_like(calibrated.depth,10.)
    result=suggest_mounted_cameras(SimpleNamespace(depth=depth),observations,
        np.array([[0.,0.,1.]]),budget=1)
    assert calls==['left','right']
    assert result['cameras']==['left']
    assert 'NOT acquired' in result['note']
