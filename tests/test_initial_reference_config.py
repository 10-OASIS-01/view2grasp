from pathlib import Path

import pytest
from pydantic import ValidationError

from view2grasp.config import ExperimentConfig,load_config


def test_initial_reference_is_opt_in_and_requires_source_ownership():
    assert not ExperimentConfig().initial_multiview_reference
    with pytest.raises(ValidationError,match='requires source object ownership'):
        ExperimentConfig(initial_multiview_reference=True)
    assert not ExperimentConfig().initial_multiview_part
    with pytest.raises(ValidationError,match='requires initial multiview object reference'):
        ExperimentConfig(initial_multiview_part=True)
    assert not ExperimentConfig().initial_object_material_mask
    with pytest.raises(ValidationError,match='requires initial multiview object reference'):
        ExperimentConfig(initial_object_material_mask=True)


def test_material_config_changes_only_initial_authority_and_closed_loop_budget():
    root=Path(__file__).resolve().parents[1]
    previous=load_config(root/'configs/mug-two-part-views.yaml').model_dump()
    material=load_config(root/'configs/mug-material-reference.yaml').model_dump()
    assert previous.pop('max_observations')==10 and material.pop('max_observations')==12
    assert previous.pop('initial_object_material_mask') is False
    assert material.pop('initial_object_material_mask') is True
    assert previous==material


def test_multiview_config_keeps_existing_observation_and_search_limits():
    root=Path(__file__).resolve().parents[1]
    original=load_config(root/'configs/mug-recovery-highres.yaml').model_dump()
    new=load_config(root/'configs/mug-multiview-reference.yaml').model_dump()
    assert new.pop('initial_multiview_reference') is True
    assert original.pop('initial_multiview_reference') is False
    assert new==original


def test_complete_reference_config_only_enables_explicit_part_initialization():
    root=Path(__file__).resolve().parents[1]
    previous=load_config(root/'configs/mug-multiview-reference.yaml').model_dump()
    complete=load_config(root/'configs/mug-complete-reference.yaml').model_dump()
    assert complete.pop('initial_multiview_part') is True
    assert previous.pop('initial_multiview_part') is False
    assert complete==previous


def test_second_part_view_is_bounded_and_explicitly_configured():
    root=Path(__file__).resolve().parents[1]
    one=load_config(root/'configs/mug-complete-reference.yaml').model_dump()
    two=load_config(root/'configs/mug-two-part-views.yaml').model_dump()
    assert one.pop('initial_part_views')==1 and two.pop('initial_part_views')==2
    assert one==two
    for value in (0,3):
        with pytest.raises(ValidationError):ExperimentConfig(initial_part_views=value)
    with pytest.raises(ValidationError,match='require explicit part initialization'):
        ExperimentConfig(initial_part_views=2)


def test_second_source_pose_view_keeps_total_budget_and_requires_initial_reference():
    root=Path(__file__).resolve().parents[1]
    one=load_config(root/'configs/mug-material-reference.yaml').model_dump()
    two=load_config(root/'configs/mug-ranked-recovery.yaml').model_dump()
    assert one.pop('source_pose_recovery_views')==1
    assert two.pop('source_pose_recovery_views')==2
    assert one==two and two['max_observations']==12
    for value in (0,3):
        with pytest.raises(ValidationError):ExperimentConfig(source_pose_recovery_views=value)
    with pytest.raises(ValidationError,match='require initial multiview reference'):
        ExperimentConfig(source_pose_recovery_views=2)
