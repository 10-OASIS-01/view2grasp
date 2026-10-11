"""Abstention survives existing deletion and contact association paths."""
import numpy as np
from scipy.spatial import cKDTree
from view2grasp.mapping import DepthMap,OCCUPIED


def test_retained_observation_survives_legacy_radius_retirement_and_target_halo():
    map_=DepthMap()
    target=np.array([[0.,0.,1.]])
    abstained=np.array([[.003,0.,1.]])
    neighbor=np.array([[.0062,0.,1.]])
    map_.surface=np.vstack([target,abstained,neighbor]);map_.tree=cKDTree(map_.surface)
    map_.set_target(target)
    # Before retention, the old 4 mm target halo would hide the 3 mm endpoint.
    map_.retain_observed_endpoints(np.vstack([abstained,neighbor]))
    assert not map_.contact_allowed(abstained).any()
    map_.remove_target_endpoints(target)
    assert np.array_equal(map_.surface,np.vstack([abstained,neighbor]))
    assert np.all(map_.classify(abstained)==OCCUPIED)
    assert not map_.contact_allowed(abstained).any()


def test_retention_does_not_insert_an_occupancy_or_free_space_certificate():
    map_=DepthMap()
    before=map_.classify(np.array([[0.,0.,1.]]))
    map_.retain_observed_endpoints(np.array([[0.,0.,1.]]))
    assert len(map_.surface)==0
    assert np.array_equal(map_.classify(np.array([[0.,0.,1.]])),before)
    map_.surface=np.array([[.00001,0.,1.]])
    map_.tree=cKDTree(map_.surface)
    map_.remove_target_endpoints(map_.surface.copy())
    assert len(map_.surface)==0  # 10 um is not numerical source equality.


def test_clear_dynamic_ends_the_whole_observation_epoch():
    map_=DepthMap();point=np.array([[0.,0.,1.]])
    map_.surface=point.copy();map_.tree=cKDTree(map_.surface)
    map_.retain_observed_endpoints(point)
    map_.clear_dynamic()
    assert len(map_.surface)==len(map_.retained_endpoints)==0
    assert map_.views==[] and map_.retained_endpoint_tree is None
