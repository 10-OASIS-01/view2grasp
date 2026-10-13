"""A retired payload still occludes camera advice, without changing the map."""
from types import SimpleNamespace
import numpy as np

from test_geometry import observation
from test_runner import _budget_episode
from view2grasp.view_advisor import calibrated_visibility


def test_postclose_runner_passes_payload_surface_to_advice_without_restoring_static_map(tmp_path,monkeypatch):
    result,evidence=_budget_episode(tmp_path,monkeypatch,recovery_views=1,planned_on=2,
        max_observations=6,retire_payload_from_surface=True)
    post=[item for item in evidence.advice_inputs if len(item['static_surface'])==1]
    assert len(post)==3 and result['observations']==6
    payload=np.array([[i*.001,0.,1.] for i in range(20)])
    for item in post:
        assert np.array_equal(item['observed_points'],np.vstack([item['static_surface'],payload]))
    assert np.array_equal(evidence.map.surface,[[.8,.9,1.]])
    assert np.all(evidence.map.classify(payload)==0)  # No UNKNOWN permission from advice.
    logged=[item['occlusion_evidence'] for item in evidence.events if item['kind']=='postgrasp_view_advice']
    assert len(logged)==3
    assert all(item['payload_reference_points']==20 and item['static_surface_points']==1 for item in logged)
    assert all(item['visibility_prediction_tolerance_unchanged'] is True for item in logged)


def test_measured_retired_surface_blocks_visibility_with_existing_prediction_tolerance():
    obs=observation();robot=SimpleNamespace(depth=lambda _:np.full_like(obs.depth,10.))
    queries=np.array([[0.,0.,1.02],[0.,0.,1.007]])
    static=np.array([[.5,.5,1.]])  # Does not cover the query pixel.
    payload=np.array([[0.,0.,1.]])
    before=static.copy();retired=payload.copy()
    assert calibrated_visibility(robot,obs,queries,static).tolist()==[True,True]
    assert calibrated_visibility(robot,obs,queries,np.vstack([static,payload])).tolist()==[False,True]
    # Advice's unchanged +8 mm sparse-surface tolerance remains an estimate,
    # never a statement that either query has been measured free by the map.
    assert np.array_equal(static,before) and np.array_equal(payload,retired)
