from pathlib import Path

import pytest
from pydantic import ValidationError

from view2grasp.config import ExperimentConfig,load_config


def test_initial_reference_is_opt_in_and_requires_source_ownership():
    assert not ExperimentConfig().initial_multiview_reference
    with pytest.raises(ValidationError,match='requires source object ownership'):
        ExperimentConfig(initial_multiview_reference=True)


def test_multiview_config_keeps_existing_observation_and_search_limits():
    root=Path(__file__).resolve().parents[1]
    original=load_config(root/'configs/mug-recovery-highres.yaml').model_dump()
    new=load_config(root/'configs/mug-multiview-reference.yaml').model_dump()
    assert new.pop('initial_multiview_reference') is True
    assert original.pop('initial_multiview_reference') is False
    assert new==original
