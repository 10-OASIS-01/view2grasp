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
