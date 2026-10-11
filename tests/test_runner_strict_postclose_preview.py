"""Runner spends existing later post-close slots on strict measured refusals."""
import numpy as np
import pytest

from view2grasp.postclose_preview import LiftPreview
from test_runner_mug_postclose_contact import run_case


def test_first_postclose_view_unchanged_later_views_use_current_strict_evidence(tmp_path,monkeypatch):
    robot_unknown=np.array([[.7,.2,1.4]])
    payload_unknown=np.array([[.2,.1,1.6]])
    def preview(e,planner,evidence,start,gripper):
        assert e.sim.phase=='close' and planner.attached is None
        assert evidence.semantic_contact_only
        assert len(evidence.current_ownership)==e.sim.frame-1
        assert len(evidence.contact_points)==len(e.measured)
        # Registration failed in this fixture: historical source ownership is
        # retained as evidence but cannot be attached or retired as a prediction.
        assert evidence.registered_source_points is None
        return LiftPreview(robot_unknown,payload_unknown,
            {'mapping_evidence_available':True,'occupied_unresolved':True,'incomplete':True},
            e.map.surface.copy())
    e=run_case(tmp_path,monkeypatch,strict_preview_override=preview)
    assert [p['frame'] for p in e.previews]==[2,3]
    assert all(p['steps']==e.physical_steps and p['phase']=='close' for p in e.previews)
    advice=[a for a in e.advice if a['phase']=='close']
    assert len(advice)==3
    assert not np.array_equal(advice[0]['required'],robot_unknown)
    for item in advice[1:]:
        assert np.array_equal(item['required'],robot_unknown)
        assert np.array_equal(np.unique(item['coverage'],axis=0),payload_unknown)
    logged=[x for x in e.events if x['kind']=='postgrasp_view_advice']
    assert 'strict_lift_evidence' not in logged[0]
    assert all(x['strict_lift_evidence']['occupied_unresolved'] for x in logged[1:])
    assert e.final_lift_checks==[{'contact':True,'connected':True}]
    assert e.result['observations']==4  # Unchanged configured episode cap.
    assert e.retired==[2,3,4]  # Previews never retire endpoints on the live map.


@pytest.mark.parametrize('reason',['occupied','IK','deadline','complete_nominal'])
def test_no_actual_unknown_does_not_spend_view_or_bypass_final_planner(tmp_path,monkeypatch,reason):
    def preview(e,*args):
        return LiftPreview(np.empty((0,3)),np.empty((0,3)),
            {'mapping_evidence_available':False,'occupied_unresolved':reason=='occupied',
                'incomplete':reason!='complete_nominal','failure':reason},e.map.surface.copy())
    e=run_case(tmp_path,monkeypatch,strict_preview_override=preview,expected_observations=2)
    assert len(e.previews)==1
    assert len([a for a in e.advice if a['phase']=='close'])==1
    assert len(e.requests)==1 and e.requests[0]['frame']==2
    assert e.final_lift_checks==[{'contact':True,'connected':True}]
    assert e.result['failure']=='planning: lift strict test lift stop'
    assert e.retired==[2]


def test_payload_only_unknown_does_not_reintroduce_raw_robot_sweep(tmp_path,monkeypatch):
    payload_unknown=np.array([[.4,.1,1.3]])
    def preview(e,*args):
        return LiftPreview(np.empty((0,3)),payload_unknown,
            {'mapping_evidence_available':True,'occupied_unresolved':False,'incomplete':True},
            e.map.surface.copy())
    e=run_case(tmp_path,monkeypatch,strict_preview_override=preview)
    advice=[a for a in e.advice if a['phase']=='close']
    for item in advice[1:]:
        assert item['required'] is None
        assert np.array_equal(np.unique(item['coverage'],axis=0),payload_unknown)


def test_default_mug_never_calls_opt_in_strict_preview(tmp_path,monkeypatch):
    def forbidden(*args):
        pytest.fail('default mug must preserve its previous view policy')
    e=run_case(tmp_path,monkeypatch,ownership=False,strict_preview_override=forbidden)
    assert not e.previews
    assert e.result['observations']==4
