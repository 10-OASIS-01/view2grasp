"""Rigid collision hypotheses neither invent observations nor grant clearance."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from view2grasp.payload_registration import PayloadRegistrationError, register_payload_source


def cloud():
    rng = np.random.default_rng(241)
    return rng.uniform([-.045,-.065,-.04],[.045,.065,.04],(160,3)) + [0,0,1.]


def known_motion(points):
    rotation = Rotation.from_rotvec([.035,-.05,.025]).as_matrix()
    center = points.mean(0)
    translation = center + [.004,-.003,.002] - rotation @ center
    return points @ rotation.T + translation, rotation, translation


def test_known_rigid_motion_keeps_entire_source_and_does_not_mutate_inputs():
    reference = cloud(); current, rotation, translation = known_motion(reference)
    source = np.vstack([reference[::7], reference.mean(0)+[.01,.015,-.02]])
    snapshots = [x.copy() for x in (reference,current,source)]
    result = register_payload_source(reference,current,source)
    assert np.allclose(result.R,rotation,atol=1e-7)
    assert np.allclose(result.t,translation,atol=1e-7)
    assert np.allclose(result.transformed_source_points,source @ rotation.T+translation,atol=1e-7)
    assert len(result.transformed_source_points)==len(source)
    assert result.report['points_pruned_from_source']==0
    assert result.report['selected']['source_coverage']==1.
    for field in ('grants_contact_permission','grants_free_space','grants_source_certificate','map_modified'):
        assert result.report[field] is False
    for actual,before in zip((reference,current,source),snapshots):assert np.array_equal(actual,before)
    with pytest.raises(ValueError):result.transformed_source_points[0]=0
    with pytest.raises(ValueError):result.R[0]=0
    json.dumps(result.report)


def test_majority_measured_overlap_can_register_without_dropping_unmatched_source():
    reference=cloud();full,rotation,translation=known_motion(reference)
    # Missing measurements remain missing; the complete source hypothesis still
    # has every input point. Distant unrelated returns do not become source.
    current=np.vstack([full[:120],full[:20]+[.2,0,0]])
    result=register_payload_source(reference,current,reference)
    assert len(result.transformed_source_points)==len(reference)
    assert np.allclose(result.R,rotation,atol=1e-7)
    assert np.allclose(result.t,translation,atol=1e-7)
    assert .5<=result.report['selected']['source_coverage']<1.


@pytest.mark.parametrize('fault',['far_motion','minority_overlap','line','plane'])
def test_insufficient_overlap_or_degenerate_geometry_never_returns_a_transform(fault):
    reference=cloud();current=reference.copy()
    if fault=='far_motion':current=reference+[.12,0,0]
    elif fault=='minority_overlap':current=np.vstack([reference[:32],reference[32:]+[.2,0,0]])
    elif fault=='line':
        reference=np.c_[np.linspace(-.06,.06,160),np.zeros(160),np.ones(160)];current=reference.copy()
    elif fault=='plane':
        x,y=np.meshgrid(np.linspace(-.06,.06,13),np.linspace(-.06,.06,13))
        reference=np.c_[x.ravel(),y.ravel(),np.ones(x.size)];current=reference.copy()
    with pytest.raises(PayloadRegistrationError) as caught:
        register_payload_source(reference,current,reference)
    assert caught.value.report['accepted'] is False
    assert caught.value.report['failure'] in ('insufficient_bounded_overlap','geometric_degeneracy','ambiguous_local_pose')
    assert not caught.value.report['grants_free_space']


def test_symmetric_object_rejects_locally_equivalent_poses_for_complete_source():
    # A finite cylinder is invariant to one angular sample step. Several local
    # fits explain the measured set, but move the same source point differently.
    angle=np.arange(24)*np.pi/12
    rings=[np.c_[.04*np.cos(angle),.04*np.sin(angle),np.full(24,z)] for z in np.linspace(.95,1.05,7)]
    reference=np.vstack(rings)
    with pytest.raises(PayloadRegistrationError) as caught:
        register_payload_source(reference,reference,reference)
    assert caught.value.report['failure']=='ambiguous_local_pose'
    assert caught.value.report['maximum_source_disagreement_m']>.004


@pytest.mark.parametrize('fault',['nan','small','bad_parameters'])
def test_invalid_inputs_fail_before_registration(fault):
    reference=cloud();current=reference.copy();kwargs={}
    if fault=='nan':current[0,0]=np.nan
    elif fault=='small':current=current[:4]
    else:kwargs['minimum_inlier_points']=True
    with pytest.raises(PayloadRegistrationError) as caught:
        register_payload_source(reference,current,reference,**kwargs)
    assert caught.value.report['failure'] in ('invalid_input_cloud','invalid_parameters')


def test_unchanged_mug12_clouds_have_a_bounded_nondegenerate_local_fit():
    fixture=np.load(Path(__file__).parent/'fixtures/mug_payload_registration.npz')
    metadata=json.loads(str(fixture['metadata']))
    original,current,source=[fixture[key] for key in ('original_reference','current_object_points','source_owned_points')]
    assert [len(original),len(current),len(source)]==[1471,2313,458]
    result=register_payload_source(original,current,source)
    report=result.report
    assert report['accepted']
    assert report['selected']['source_coverage']>=.5
    assert report['maximum_source_disagreement_m']<=.004
    assert report['point_plane_condition']<=100
    assert min(report['inlier_coordinate_singular_m'])>=.004
    assert len(result.transformed_source_points)==458
    assert np.allclose(result.transformed_source_points,source @ result.R.T+result.t)
    assert report['input_sha256']['original_reference'] != report['input_sha256']['current_object_points']
    assert metadata['episode_success_claimed'] is False
    assert 'legacy tracker' in metadata['current_cloud_semantics']
