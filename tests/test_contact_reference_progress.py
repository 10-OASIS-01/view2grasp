"""Fewer measured boundary violations may warrant one last strict correction."""
import copy
import json
from pathlib import Path

import pytest

from view2grasp.runner import _contact_annotation_progressed


def actual_pair():
    fixture=json.loads((Path(__file__).parent/'fixtures/contact_reference_progress.json').read_text())
    return fixture['first'],fixture['second']


def test_saved_occluded13_frame4_reduces_real_rejections_despite_anchor_loss():
    previous,current=actual_pair()
    assert [(x['anchor_pixels'],x['valid_pixels'],x['rejected_pixel_count'])
        for x in (previous,current)]==[(72,649,32),(56,580,1)]
    snapshots=copy.deepcopy((previous,current))
    assert _contact_annotation_progressed(previous,current)
    assert (previous,current)==snapshots  # No diagnostic/mask/pixel repair.
    # The last wrong pixel still exists and must be rejected by the resolver.
    assert current['failure']=='reference_inconsistency'
    assert current['rejection_counts']['reference_distance']==1
    assert any('reference_distance' in sample['reasons'] for sample in current['rejected_samples'])
    assert not _contact_annotation_progressed(current,current)


@pytest.mark.parametrize('role',['object','grasp_part'])
def test_new_reference_progress_requires_same_role_but_supports_both(role):
    previous,current=actual_pair();previous['role']=current['role']=role
    assert _contact_annotation_progressed(previous,current)


@pytest.mark.parametrize('fault',[
    'same_count','more_distance','same_total','zero_remaining','zero_anchor','few_anchors',
    'neighbor_component','neighbor_count','overlap_failure','different_frame','different_camera',
    'reverse_role','different_shape','missing_distance','missing_valid','bool_distance',
    'float_distance','negative_distance','bool_total','negative_total','bool_anchor',
    'negative_anchor','bool_valid','float_valid','zero_valid','valid_over_proposed',
    'proposed_over_image','anchors_plus_distance_over_valid','invalid_counts_container',
    'bool_frame','missing_frame','missing_camera','empty_camera','invalid_role','bool_shape',
    'nonzero_overlap_count','bool_component_count'])
def test_reference_progress_does_not_unlock_for_stale_neighbors_or_bad_counts(fault):
    previous,current=actual_pair()
    if fault=='same_count':current['rejection_counts']['reference_distance']=32
    elif fault=='more_distance':current['rejection_counts']['reference_distance']=33
    elif fault=='same_total':current['rejected_pixel_count']=32
    elif fault=='zero_remaining':current['rejected_pixel_count']=0;current['rejection_counts']['reference_distance']=0
    elif fault=='zero_anchor':current['anchor_pixels']=0
    elif fault=='few_anchors':current['anchor_pixels']=3
    elif fault=='neighbor_component':current['failure']='unanchored_component'
    elif fault=='neighbor_count':current['rejection_counts']['unanchored_component']=1
    elif fault=='overlap_failure':current['failure']='insufficient_filtered_depth'
    elif fault=='different_frame':current['frame_id']+=1
    elif fault=='different_camera':current['camera_id']='other'
    elif fault=='reverse_role':previous['role']='grasp_part'
    elif fault=='different_shape':current['source_shape_hw']=[512,512]
    elif fault=='missing_distance':current['rejection_counts'].pop('reference_distance')
    elif fault=='missing_valid':current.pop('valid_pixels')
    elif fault=='bool_distance':current['rejection_counts']['reference_distance']=True
    elif fault=='float_distance':current['rejection_counts']['reference_distance']=1.
    elif fault=='negative_distance':current['rejection_counts']['reference_distance']=-1
    elif fault=='bool_total':current['rejected_pixel_count']=True
    elif fault=='negative_total':current['rejected_pixel_count']=-1
    elif fault=='bool_anchor':current['anchor_pixels']=True
    elif fault=='negative_anchor':current['anchor_pixels']=-1
    elif fault=='bool_valid':current['valid_pixels']=True
    elif fault=='float_valid':current['valid_pixels']=580.
    elif fault=='zero_valid':current['valid_pixels']=0
    elif fault=='valid_over_proposed':current['valid_pixels']=current['proposed_pixels']+1
    elif fault=='proposed_over_image':current['proposed_pixels']=1024*1024+1
    elif fault=='anchors_plus_distance_over_valid':current['valid_pixels']=56
    elif fault=='invalid_counts_container':current['rejection_counts']=[]
    elif fault=='bool_frame':previous['frame_id']=current['frame_id']=True
    elif fault=='missing_frame':current.pop('frame_id')
    elif fault=='missing_camera':current.pop('camera_id')
    elif fault=='empty_camera':previous['camera_id']=current['camera_id']=''
    elif fault=='invalid_role':previous['role']=current['role']='background'
    elif fault=='bool_shape':current['source_shape_hw']=[True,1024]
    elif fault=='nonzero_overlap_count':current['rejection_counts']['reference_overlap_insufficient']=1
    elif fault=='bool_component_count':current['rejection_counts']['unanchored_component']=False
    assert not _contact_annotation_progressed(previous,current)


@pytest.mark.parametrize('before_total,before_valid,after_total,after_valid',[
    (100,1000,90,100),  # Fewer bad pixels, but a much worse observed surface.
    (32,640,1,20),     # Equal rejection ratio is not new geometric progress.
])
def test_shrinking_region_alone_does_not_show_better_geometric_agreement(
        before_total,before_valid,after_total,after_valid):
    previous,current=actual_pair()
    for report,total,valid in ((previous,before_total,before_valid),(current,after_total,after_valid)):
        report['anchor_pixels']=4
        report['rejected_pixel_count']=report['rejection_counts']['reference_distance']=total
        report['valid_pixels']=report['proposed_pixels']=valid
    # A lower anchor count avoids the preserved legacy equal-anchor rule and
    # isolates the new reference-boundary path under review.
    previous['anchor_pixels']=5
    assert not _contact_annotation_progressed(previous,current)


@pytest.mark.parametrize('previous,current',[(None,{}),({},{}),([],{}),({},'bad')])
def test_malformed_diagnostics_fail_closed_instead_of_raising(previous,current):
    assert not _contact_annotation_progressed(previous,current)
