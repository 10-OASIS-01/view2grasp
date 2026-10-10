"""Explicit process boundary for incompatible simulator dependency locks."""
from pathlib import Path
import os
import subprocess


def mobile_process(args):
    root=Path(__file__).resolve().parents[2]
    python=root/'runtimes/robocasa/.venv/bin/python'
    if not python.exists():raise RuntimeError('Run scripts/setup_robocasa.py first')
    env=os.environ.copy()
    env['PYTHONPATH']=str(root/'src')
    return subprocess.run([str(python),'-m','view2grasp.mobile_cli',*map(str,args)],env=env,check=True)
