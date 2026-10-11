import json
import pytest
from typer.testing import CliRunner
from pydantic import ValidationError
from view2grasp.cli import app
from view2grasp.config import ExperimentConfig,load_config,fingerprint
from view2grasp.evaluation import wilson,failure_category


def test_config_is_explicit_and_bounded():
    config=load_config('configs/default.yaml')
    assert config==ExperimentConfig()
    assert len(config.test_seeds)==20 and min(config.test_seeds)>=100
    assert fingerprint(config)==fingerprint(config)
    assert ExperimentConfig(max_observations=9).max_observations==9
    assert ExperimentConfig(max_observations=10).max_observations==10
    with pytest.raises(ValidationError):ExperimentConfig(max_observations=11)
    with pytest.raises(ValidationError):ExperimentConfig(unknown=True)


def test_empty_development_summary_is_not_zero_percent_benchmark(tmp_path):
    result=CliRunner().invoke(app,['summarize','--root',str(tmp_path)])
    assert result.exit_code==0
    data=json.loads((tmp_path/'summary.json').read_text())
    assert data['expected'] is None and data['kind']=='development'
    assert data['completed']==0
    assert all(g['success_rate'] is None for g in data['groups'])


def test_metrics_and_failure_taxonomy():
    lo,hi=wilson(0,20);assert abs(lo)<1e-8 and .15<hi<.17
    assert failure_category('planning: unknown space')=='mapping'
    assert failure_category('planning: grounding: depth edge')=='grounding'
    assert failure_category('execution: joint tracking error')=='execution'


def test_small_validation_has_15_trials_and_dry_run_does_not_write(tmp_path):
    root=tmp_path/'benchmark'
    result=CliRunner().invoke(app,['evaluate','--dry-run','--root',str(root)])
    assert result.exit_code==0,result.output
    data=json.loads(result.output)
    assert len(data['episodes'])==15 and len(data['tasks'])==5
    assert data['kind']=='small_development_validation' and data['strategies']==['active']
    assert not root.exists()


def test_large_strategy_sweep_is_rejected_before_any_run(tmp_path):
    result=CliRunner().invoke(app,['evaluate','--strategy','all','--root',str(tmp_path/'benchmark')])
    assert result.exit_code!=0 and '15 episodes total' in result.output
    assert not (tmp_path/'benchmark').exists()


def test_missing_metrics_remain_null_instead_of_fake_zero(tmp_path):
    episode=tmp_path/'NavigateKitchen_active_0003';episode.mkdir()
    (episode/'result.json').write_text(json.dumps(dict(task='NavigateKitchen',strategy='active',online=True,
        success=False,collision=False,attempts=0,failure='planning: unknown space',official_success=None,path_length_m=None)))
    result=CliRunner().invoke(app,['summarize','--root',str(tmp_path)])
    assert result.exit_code==0,result.output
    row=next(r for r in json.loads((tmp_path/'summary.json').read_text())['groups'] if r['task']=='NavigateKitchen' and r['strategy']=='active')
    assert row['official_success_rate'] is None and row['mean_path_length_m'] is None


def test_unmeasured_grasps_do_not_count_as_failed_attempts(tmp_path):
    from dataclasses import asdict
    from view2grasp.types import EpisodeRecord

    records=[EpisodeRecord('DeliverCanToBowl',seed,'active') for seed in (3,4,5)]
    assert records[0].first_grasp_success is None
    records[1].first_grasp_success=True
    records[1].attempts=1
    records[2].first_grasp_success=False
    records[2].attempts=1
    for record in records:
        episode=tmp_path/f'{record.task}_{record.strategy}_{record.seed:04d}'
        episode.mkdir()
        (episode/'result.json').write_text(json.dumps(asdict(record)))
    result=CliRunner().invoke(app,['summarize','--root',str(tmp_path)])
    assert result.exit_code==0,result.output
    row=next(r for r in json.loads((tmp_path/'summary.json').read_text())['groups']
             if r['task']=='DeliverCanToBowl' and r['strategy']=='active')
    assert row['n']==3
    assert row['first_grasp_success_rate']==.5
