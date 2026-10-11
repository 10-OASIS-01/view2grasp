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


def semantic_example():
    obs=observation()
    obs=replace(obs,rgb=np.zeros((512,512,3),np.uint8),depth=np.ones((512,512)),
        K=np.array([[365.,0,255.5],[0,365.,255.5],[0,0,1.]]))
    x,y=np.meshgrid(np.linspace(-.025,.025,15),np.linspace(-.025,.025,15))
    return obs,np.c_[x.ravel()-.12,y.ravel(),np.full(x.size,.95)]


def test_semantic_focus_prefers_distinct_pixels_over_same_visible_point_fraction(monkeypatch):
    import view2grasp.view_advisor as advisor
    obs,semantic=semantic_example();payload=np.array([[0.,0.,1.]])
    monkeypatch.setattr(advisor,'_visible',lambda robot,obs,points,view,observed_points=None:np.ones(len(points),bool))
    old=advisor.suggest_views(None,obs,payload,budget=1,focus_points=semantic)
    new=advisor.suggest_views(None,obs,payload,budget=1,focus_points=semantic,semantic_focus=semantic)
    old_quality=advisor._semantic_projection(obs,semantic,old['views'][0],np.ones(len(semantic),bool))
    quality=new['predicted_semantic_focus'][0]
    assert old['predicted_coverage']==new['predicted_coverage']==1.
    assert old['predicted_target_visibility']==new['predicted_target_visibility']==1.
    assert quality['unique_visible_pixels']>old_quality['unique_visible_pixels']
    assert quality['minor_axis_span_pixels']>old_quality['minor_axis_span_pixels']
    assert 'not semantic verification' in new['semantic_focus_note']


def test_semantic_focus_cannot_sacrifice_required_or_payload_coverage(monkeypatch):
    import view2grasp.view_advisor as advisor
    obs,semantic=semantic_example();payload=np.zeros((2,3));required=np.ones((2,3))
    for hard_priority in ('required','payload'):
        def visible(robot,obs,points,view,observed_points=None):
            if points is semantic:return np.ones(len(points),bool)
            prioritized=points is (required if hard_priority=='required' else payload)
            return np.array([True,view[:2]==(-180.,-10.)]) if prioritized else np.ones(len(points),bool)
        monkeypatch.setattr(advisor,'_visible',visible)
        advice=advisor.suggest_views(None,obs,payload,budget=1,required_points=required,semantic_focus=semantic)
        assert advice['views']==[(-180.,-10.,1.3)]
        assert advice['predicted_coverage']==advice['predicted_required_coverage']==1.


def test_semantic_only_view_still_uses_existing_image_geometry_without_mapping_need(monkeypatch):
    import view2grasp.view_advisor as advisor
    obs,semantic=semantic_example();calls=[]
    def visible(robot,obs,points,view,observed_points=None):
        calls.append(view);return np.ones(len(points),bool)
    monkeypatch.setattr(advisor,'_visible',visible)
    advice=advisor.suggest_views(None,obs,np.empty((0,3)),budget=1,semantic_focus=semantic)
    assert len(advice['views'])==1 and advice['unknown_points']==0
    assert advice['predicted_coverage']==1.
    assert advice['predicted_semantic_focus'][0]['unique_visible_pixels']>0
