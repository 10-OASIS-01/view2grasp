from dataclasses import replace
import hashlib,json

import numpy as np
import pytest

from test_geometry import observation
from view2grasp.annotation_diagnostic import build_annotation_diagnostic


def feedback(obs, pixels=((0,0),(15,16),(31,31))):
    h,w=obs.depth.shape
    return {'frame_id':obs.frame_id,'camera_id':obs.camera_id,
        'coordinate_space':'source_image_normalized_pixel_centers','source_shape_hw':[h,w],
        'rejected_samples':[{'pixel':list(p),'normalized':[p[0]/(w-1),p[1]/(h-1)]} for p in pixels]}


def test_centers_and_raw_image_stay_exact_even_under_marks_and_grid_crossings():
    obs=observation();obs.rgb[:]=np.random.default_rng(9).integers(0,256,obs.rgb.shape,dtype=np.uint8)
    before=obs.rgb.copy();diagnostic=build_annotation_diagnostic(obs,feedback(obs))
    x0,y0=diagnostic.content_offset_xy;scale=diagnostic.scale;h,w=obs.depth.shape
    centers=diagnostic.rgb[y0::scale,x0::scale]
    assert np.array_equal(centers,before) and np.array_equal(obs.rgb,before)
    for x,y in diagnostic.rejected_pixels:
        edge_x=x*scale+(scale//2 if x<w-1 else -scale//2)
        assert np.array_equal(diagnostic.rgb[y*scale,edge_x],[220,0,160])
    assert not diagnostic.rgb.flags.writeable
    second=build_annotation_diagnostic(obs,feedback(obs))
    assert np.array_equal(diagnostic.rgb,second.rgb)
    p=diagnostic.provenance()
    assert json.loads(json.dumps(p,allow_nan=False))==p
    assert p['source_rgb_sha256']==hashlib.sha256(before.tobytes()).hexdigest()
    assert p['diagnostic_rgb_sha256']==hashlib.sha256(diagnostic.rgb.tobytes()).hexdigest()
    assert not p['new_observation'] and p['source_pixel_centers_unchanged']
    assert p['rejected_pixels'][1]['pixel']==[15,16]
    assert 'NOT object labels' in p['scope']
    assert p['normalized_pixel_centers_match_primary'] and not p['external_margins_or_legend']
    assert p['content_bounds_xyxy']==[0,0,(w-1)*scale+1,(h-1)*scale+1]


def test_current_crop_coordinates_do_not_use_retained_source_pixel_indices():
    obs=observation();f=feedback(obs,((17,12),))
    f.update(coordinate_space='supplied_crop_normalized_pixel_centers',image_shape_hw=[32,32],source_shape_hw=[512,512])
    f['rejected_samples'][0]['source_pixel']=[215,247]
    d=build_annotation_diagnostic(obs,f)
    assert d.rejected_pixels==((17,12),)
    assert d.provenance()['source_shape_hw']==[32,32]


@pytest.mark.parametrize('fault', ['stale','camera','coordinate_space','shape','nonfinite','outside','too_many'])
def test_bad_feedback_fails_before_a_diagnostic_image_can_be_used(fault):
    obs=observation();f=feedback(obs)
    if fault=='stale':f['frame_id']=99
    elif fault=='camera':f['camera_id']='other'
    elif fault=='coordinate_space':f['coordinate_space']='ambiguous'
    elif fault=='shape':f['source_shape_hw']=[512,512]
    elif fault=='nonfinite':f['rejected_samples'][0]['normalized']=[np.nan,.5]
    elif fault=='outside':f['rejected_samples'][0]['normalized']=[1.1,.5]
    else:f['rejected_samples']*=5
    with pytest.raises(ValueError):build_annotation_diagnostic(obs,f)


@pytest.mark.parametrize('fault', ['frame','camera','pixels','shape'])
def test_diagnostic_cannot_be_attached_to_another_image(fault):
    obs=observation();d=build_annotation_diagnostic(obs,feedback(obs))
    if fault=='frame':obs=replace(obs,frame_id=99)
    elif fault=='camera':obs=replace(obs,camera_id='other')
    elif fault=='shape':obs=replace(obs,rgb=np.zeros((16,16,3),np.uint8))
    else:obs.rgb[5,5]=1
    with pytest.raises(ValueError,match='current unmarked image'):d.validate_source(obs)


def test_empty_or_duplicate_samples_do_not_invent_rejected_pixels():
    obs=observation()
    assert not build_annotation_diagnostic(obs,feedback(obs,())).rejected_pixels
    assert build_annotation_diagnostic(obs,feedback(obs,((8,8),(8,8)))).rejected_pixels==((8,8),)


@pytest.mark.parametrize('shape',[(2,2),(17,31),(128,128)])
@pytest.mark.parametrize('scale',[4,5,6,7,8])
def test_all_corners_edges_and_grid_crossings_keep_source_centers_and_normalized_coordinates(shape,scale):
    h,w=shape
    obs=replace(observation(),rgb=np.random.default_rng(17).integers(0,256,(h,w,3),dtype=np.uint8),
        depth=np.ones((h,w)))
    rejected=((0,0),(w-1,0),(0,h-1),(w-1,h-1),(w//2,0),(w//2,h-1),
        (0,h//2),(w-1,h//2),(min(8,w-1),min(8,h-1)))
    diagnostic=build_annotation_diagnostic(obs,feedback(obs,rejected),scale=scale)
    dh,dw=diagnostic.rgb.shape[:2]
    assert (dh,dw)==((h-1)*scale+1,(w-1)*scale+1)
    assert diagnostic.content_offset_xy==(0,0)
    assert np.array_equal(diagnostic.rgb[::scale,::scale],obs.rgb)
    assert np.array_equal(np.arange(w)*scale/(dw-1),np.arange(w)/(w-1))
    assert np.array_equal(np.arange(h)*scale/(dh-1),np.arange(h)/(h-1))
    assert diagnostic.provenance()['source_to_diagnostic_pixel_centers']=={
        'scale_xy':[scale,scale],'offset_xy':[0,0]}


def test_unmarked_non_grid_pixels_are_only_endpoint_aligned_nearest_neighbor_samples():
    obs=observation();obs.rgb[:]=np.random.default_rng(21).integers(0,256,obs.rgb.shape,dtype=np.uint8)
    diagnostic=build_annotation_diagnostic(obs,feedback(obs,()),scale=4,grid_step=1000)
    dh,dw=diagnostic.rgb.shape[:2]
    expected=obs.rgb[np.rint(np.arange(dh)/4).astype(int)[:,None],
        np.rint(np.arange(dw)/4).astype(int)[None,:]]
    assert np.array_equal(diagnostic.rgb,expected)
