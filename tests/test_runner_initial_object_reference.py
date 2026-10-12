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
    assert e.registration_references
    assert all(np.array_equal(points,reference) for points in e.registration_references)
    assert all(not _contains(points,e.current_points) for points in e.registration_references)
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


def test_complementary_initial_part_keeps_ray_and_fits_all_static_sources(tmp_path,monkeypatch):
    result,e=run_lifecycle(tmp_path,monkeypatch,initialize_reference=True,initialize_part=True,max_observations=6)
    assert len(e.initializations)==2 and result['attempts']==1
    assert [(a,b) for a,b,_ in e.initializations]==[(1,3),(2,3)]
    assert [f for f,_ in e.retired]==[1,2,4,5,6]
    part=np.load(tmp_path/'mug_active_0003/initial_permitted_part_reference.npy')
    assert np.array_equal(part,np.unique(np.vstack([e.original,e.extra_initial_part]),axis=0))
    assert all(np.array_equal(points,e.original) for _,_,points in e.initializations)
    assert np.array_equal(e.candidate.approach,e.original_ray)
    assert not e.map.contact_allowed(e.predicted).any()
    assert _contains(e.source_certificates[-1],e.source_points)
    assert len(e.planner.registered_payload_local)==2*len(e.predicted)
    reference=np.load(tmp_path/'mug_active_0003/initial_ownership_reference.npy')
    assert all(np.array_equal(points,reference) for points in e.registration_references)


def test_part_failure_stops_before_any_action_and_does_not_use_primary_success_as_fallback(tmp_path,monkeypatch):
    result,e=run_lifecycle(tmp_path,monkeypatch,initialize_reference=True,initialize_part=True,
        part_initialization_ok=False,max_observations=6)
    assert result['attempts']==0 and len(e.initializations)==2
    assert not e.executed and not e.annotations and not e.registrations and not e.retired


def test_extra_initial_source_has_no_retirement_or_certificate_without_current_fit(tmp_path,monkeypatch):
    _,e=run_lifecycle(tmp_path,monkeypatch,initialize_reference=True,initialize_part=True,
        registration_ok=False,max_observations=6)
    assert [f for f,_ in e.retired]==[4,5,6]
    assert e.planner.registered_payload_local is None
    assert not _contains(e.source_certificates[-1],e.source_points)


def test_two_complementary_views_keep_the_original_side_envelope_and_all_source_clouds(tmp_path,monkeypatch):
    result,e=run_lifecycle(tmp_path,monkeypatch,initialize_reference=True,initialize_part=True,
        initialize_part_count=2,max_observations=7)
    assert result['observations']==7 and result['attempts']==1
    assert [(a,b) for a,b,_ in e.initializations]==[(1,4),(2,4),(3,4)]
    assert [f for f,_ in e.retired]==[1,2,3,5,6,7]
    assert len(e.planner.registered_payload_local)==3*len(e.predicted)
    reference=np.load(tmp_path/'mug_active_0003/initial_ownership_reference.npy')
    assert len(reference)==len(e.original)+3*len(e.source_points)
    assert e.registration_references and all(np.array_equal(points,reference) for points in e.registration_references)
    assert np.array_equal(e.candidate.approach,e.original_ray)
    assert not e.map.contact_allowed(e.predicted).any()


def test_second_complementary_failure_cannot_execute_using_first_accepted_view(tmp_path,monkeypatch):
    result,e=run_lifecycle(tmp_path,monkeypatch,initialize_reference=True,initialize_part=True,
        initialize_part_count=2,fail_part_frame=3,max_observations=7)
    assert len(e.initializations)==3 and result['attempts']==0
    assert not e.executed and not e.registrations and not e.retired
