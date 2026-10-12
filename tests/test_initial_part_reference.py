"""Explicit static part semantics never turn an object label into contact."""
from dataclasses import replace

import numpy as np
import pytest

from view2grasp.initial_object_reference import annotate_initial_object_reference,select_initial_part_view
from test_initial_object_reference import ScriptedVLM
from test_object_ownership import scene,neighbor_scene,annotation


def reply(obs,object_mask,part_mask,seeds=((2,6),(9,6))):
    result=annotation(obs,object_mask,seeds=seeds)
    region=annotation(obs,part_mask).groundings[0].model_copy(update={'role':'grasp_part'})
    return result.model_copy(update={'groundings':[*result.groundings,region]})


def initialize(obs,reference,part,replies):
    client=ScriptedVLM(replies)
    result=annotate_initial_object_reference(client,obs,replace(obs,frame_id=obs.frame_id+1),
        reference,'mug',.9,robot_depth=np.full_like(obs.depth,10.),side_part=part)
    return result,client


def test_fresh_explicit_part_can_initialize_unanchored_face_without_whole_object_contact():
    obs,xyz,mask,reference=neighbor_scene()
    part=mask.copy();part[:,:8]=False
    result,client=initialize(obs,reference,reference,[reply(obs,mask,part)])
    assert result.ownership is not None
    assert np.array_equal(result.initial_part_mask,part)
    assert np.array_equal(result.initial_part_points,xyz[part])
    assert not np.array_equal(result.initial_part_mask,result.ownership.object_mask)
    assert not result.initial_part_points.flags.writeable and not result.initial_part_mask.flags.writeable
    report=result.attempts[-1].log['initial_part']
    assert report['part_side_anchor_pixels']==0 and not report['part_component_anchor_gate_applied']
    assert not result.ownership.report['grants_contact_permission']
    assert client.requests[0][3]['annotation_roles']==('object','grasp_part')
    assert 'No permitted part is requested' not in client.requests[0][1]
    assert 'whole-object labels never do' in client.requests[0][1]


def test_unowned_part_material_rejects_entire_reply_even_if_object_is_valid():
    obs,_,mask,reference=neighbor_scene()
    part=mask.copy();part[:,:8]=False
    bad=reply(obs,mask,part,seeds=((2,6),))
    result,client=initialize(obs,reference,reference,[bad,bad])
    assert len(client.requests)==2 and result.ownership is None and result.initial_part_points is None
    assert all(a.ownership is None for a in result.attempts)
    assert all(a.log['diagnostics']['failure']=='initial_part_outside_owned_object' for a in result.attempts)


def test_bad_part_then_fresh_valid_part_has_no_merge_or_fallback():
    obs,xyz,mask,reference=neighbor_scene()
    part=mask.copy();part[:,:8]=False
    first=reply(obs,mask,part,seeds=((2,6),));second=reply(obs,mask,part)
    result,client=initialize(obs,reference,reference,[first,second])
    assert result.attempts[0].ownership is None
    assert np.array_equal(result.initial_part_points,xyz[part])
    assert not result.report['reply_results_merged'] and len(client.requests)==2


def test_whole_object_within_its_envelope_does_not_expand_the_part_envelope():
    obs,xyz,mask,part_reference=scene(end=50)
    full_reference=xyz[mask]
    part=np.zeros_like(mask);part[6:10,44:48]=True
    bad=reply(obs,mask,part,seeds=((2,6),))
    result,client=initialize(obs,full_reference,part_reference,[bad,bad])
    assert result.ownership is None and result.initial_part_points is None
    assert result.attempts[-1].log['diagnostics']['failure']=='initial_part_outside_fixed_envelope'


@pytest.mark.parametrize('fault',['missing','identity','frame','source','points_only'])
def test_part_protocol_failure_does_not_silently_use_object_or_add_retries(fault):
    obs,_,mask,reference=scene()
    decision=reply(obs,mask,mask,seeds=((2,6),))
    part=decision.groundings[-1]
    if fault=='missing':decision.groundings.pop()
    else:
        updates={'identity':{'target_id':'neighbor'},'frame':{'frame_id':99},
            'source':{'source':'rgbd'},'points_only':{'kind':'points','points':[(.1,.4)]}}[fault]
        decision.groundings[-1]=part.model_copy(update=updates)
    result,client=initialize(obs,reference,reference,[decision])
    assert result.ownership is None and result.initial_part_points is None and len(client.requests)==1


def test_side_part_must_be_measured_and_contained_in_the_original_side_object():
    obs,xyz,mask,reference=scene()
    result,client=initialize(obs,reference,xyz[6:10,8:10].reshape(-1,3),[])
    assert not client.requests and result.ownership is None and result.initial_part_points is None
    assert result.report['failure']=='invalid_initial_evidence'


def test_part_view_selection_excludes_existing_sources_and_moved_robot_frames():
    side,_,_,reference=scene(end=8)
    alternative,_,_,_=neighbor_scene()
    alternative=replace(alternative,frame_id=9)
    first=replace(alternative,frame_id=1)
    moved=replace(alternative,frame_id=10,q=alternative.q+.01)
    own={o.frame_id:np.full_like(o.depth,10.) for o in (side,alternative,first,moved)}
    selected,report=select_initial_part_view([first,side,alternative,moved],side,reference,.9,own,excluded_frames=(1,))
    assert selected is alternative and report['selected_frame']==9
    assert [r['frame_id'] for r in report['ranked_frames']]==[9]
    assert report['new_observations']==0 and report['maximum_additional_source_images']==1


def test_part_view_selection_does_not_create_membership_or_select_filtered_background():
    side,_,_,reference=scene(end=8)
    other=replace(side,frame_id=9,depth=np.full_like(side.depth,2.))
    selected,report=select_initial_part_view([other],side,reference,.9,{9:np.full_like(side.depth,10.)})
    assert selected is None and report['selected_frame'] is None


@pytest.mark.parametrize('reference',[np.empty((0,3)),np.full((4,3),np.nan),np.zeros(4)])
def test_invalid_part_view_reference_fails_explicitly(reference):
    side,_,_,_=scene()
    with pytest.raises(ValueError):select_initial_part_view([],side,reference,.9,{})
