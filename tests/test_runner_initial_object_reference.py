"""Initial object evidence never expands finger contact or skips registration."""
import numpy as np

from test_source_ownership_lifecycle import run_lifecycle, _contains


def test_initial_reference_is_frozen_before_execution_and_remains_separate(tmp_path,monkeypatch):
    result,e=run_lifecycle(tmp_path,monkeypatch,initialize_reference=True,max_observations=5)
    assert len(e.initializations)==1 and result['attempts']==1
    assert [frame for frame,_ in e.annotations]==[3,4,5]
    reference=np.load(tmp_path/'mug_active_0003/initial_ownership_reference.npy')
    assert np.array_equal(reference,np.vstack([e.original,e.source_points]))
    assert all(np.array_equal(points,reference) for _,points in e.annotations)
    assert all(np.array_equal(points,e.original) for points in e.targets)
    assert np.array_equal(e.candidate.approach,e.original_ray)
    assert _contains(e.map.surface,e.neighbor[None])
    assert not e.map.contact_allowed(e.predicted).any()
    assert [frame for frame,_ in e.retired]==[1,3,4,5]


def test_failed_initialization_stops_before_any_execution_without_legacy_fallback(tmp_path,monkeypatch):
    result,e=run_lifecycle(tmp_path,monkeypatch,initialize_reference=True,initialization_ok=False,max_observations=5)
    assert result['attempts']==0 and len(e.initializations)==1
    assert not e.executed and not e.annotations and not e.registrations and not e.retired
    assert not (tmp_path/'mug_active_0003/initial_ownership_reference.npy').exists()


def test_missing_initial_identity_stops_before_any_execution(tmp_path,monkeypatch):
    result,e=run_lifecycle(tmp_path,monkeypatch,initialize_reference=True,initial_identity=False,max_observations=5)
    assert result['attempts']==0
    assert not e.initializations and not e.executed and not e.annotations
    assert not e.registrations and not e.retired
    assert not (tmp_path/'mug_active_0003/initial_ownership_reference.npy').exists()


def test_initial_source_without_current_registration_cannot_retire_or_supply_current_cert(tmp_path,monkeypatch):
    _,e=run_lifecycle(tmp_path,monkeypatch,initialize_reference=True,registration_ok=False,max_observations=5)
    assert [frame for frame,_ in e.retired]==[3,4,5]
    assert e.planner.registered_payload_local is None
    assert not _contains(e.source_certificates[-1],e.source_points)


def test_default_keeps_old_source_annotation_transaction(tmp_path,monkeypatch):
    _,e=run_lifecycle(tmp_path,monkeypatch)
    assert not e.initializations
    assert [frame for frame,_ in e.annotations]==[1,2,3,4]
