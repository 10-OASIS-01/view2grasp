"""Reproduce the isolated runtime without touching any existing external installation."""
from pathlib import Path
import json
import subprocess


def main():
    runtime=Path(__file__).resolve().parents[1]/'runtimes/robocasa'
    pins=json.loads((runtime/'sources.json').read_text())
    (runtime/'vendor').mkdir(exist_ok=True)
    for name,pin in pins.items():
        dest=runtime/'vendor'/name
        if not dest.exists():
            subprocess.run(['git','clone',pin['url'],str(dest)],check=True)
            subprocess.run(['git','-C',str(dest),'checkout','--detach',pin['commit']],check=True)
        actual=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
        if actual!=pin['commit']:raise RuntimeError(f'{name} differs from pin; refusing to overwrite checkout')
    subprocess.run(['uv','sync','--locked'],cwd=runtime,check=True)
    python=runtime/'.venv/bin/python'
    private=runtime/'vendor/robocasa/robocasa/macros_private.py'
    if not private.exists():subprocess.run([str(python),'-m','robocasa.scripts.setup_macros'],check=True,cwd=runtime)
    print('Runtime installed. Download upstream assets separately, then run doctor --backend robocasa.')

if __name__=='__main__':main()
