from pathlib import Path
import json
import typer

app=typer.Typer(no_args_is_help=True,help="Observation-only active-view Panda grasping")

@app.command()
def doctor(backend:str='tabletop',render:bool=True,output:Path|None=None):
    """Render RGB-D, check calibration and probe the configured VLM service."""
    if backend=='robocasa':
        from .runtime import mobile_process
        args=['doctor']+(['--render'] if render else [])+(['--output',output.resolve()] if output else [])
        mobile_process(args);return
    if backend!='tabletop':raise typer.BadParameter('backend must be tabletop or robocasa')
    import sys,platform
    import httpx,numpy as np
    import mujoco,robosuite
    from .sim import Simulator
    from .geometry import cloud,project
    from .vlm_settings import VLMSettings
    settings=VLMSettings.from_env()
    sim=Simulator("block",0,128)
    obs=sim.observe()
    points=cloud(obs,8); uv,z=project(points,obs.K,obs.T_wc)
    v,u=np.mgrid[0:128:8,0:128:8]
    error=float(np.max(np.abs(uv-np.stack([u,v],-1).reshape(-1,2))))
    r=httpx.get(settings.base_url+"/models",headers={"Authorization":"Bearer "+settings.api_key},timeout=10)
    r.raise_for_status()
    report={"python":sys.version,"platform":platform.platform(),"robosuite":robosuite.__version__,"mujoco":mujoco.__version__,
            "render_ok":True,"reprojection_max_px":error,"vlm":settings.public_metadata(),"models":r.json()}
    sim.close()
    if output:
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

@app.command('check-vlm')
def check_vlm(output:Path=Path('artifacts/vlm-check')):
    """Check a live image request and visual grounding without a simulator."""
    from .vlm_probe import run_vlm_probe
    report=run_vlm_probe(output)
    print(json.dumps(report,indent=2))
    if not report['success']:
        raise typer.Exit(code=1)

@app.command()
def run(task:str="block",seed:int=0,strategy:str="active",root:Path=Path("runs/dev"),replay:Path|None=None,config:Path|None=None,backend:str='auto'):
    """Run one online episode (or explicitly labeled response replay)."""
    if strategy not in ("active","random","fixed"): raise typer.BadParameter("unknown strategy")
    mobile_tasks=('NavigateKitchen','DeliverCanToBowl','DeliverBrewedCoffee')
    if task in mobile_tasks:
        if backend not in ('auto','robocasa'):raise typer.BadParameter('mobile task requires robocasa')
        if replay:raise typer.BadParameter('use offline state replay for mobile recordings; online response replay is not enabled')
        from .runtime import mobile_process
        args=['run','--task',task,'--seed',seed,'--strategy',strategy,'--root',root.resolve()]
        if config:args+=['--config',config.resolve()]
        mobile_process(args);return
    if task not in ("block","mug","occluded"): raise typer.BadParameter("unknown task")
    if backend not in ('auto','tabletop'):raise typer.BadParameter('tabletop task requires tabletop backend')
    from .runner import run_episode
    from .config import load_config
    print(json.dumps(run_episode(task,seed,strategy,root,replay,load_config(config)),indent=2))

@app.command()
def evaluate(root:Path=Path("runs/validation"),task:str="all",strategy:str="active",config:Path|None=None,backend:str='all',dry_run:bool=False):
    """Small development validation, at most 15 episodes; never overwrite records."""
    from .config import load_config,fingerprint
    settings=load_config(config)
    primary_tasks=("block","mug","occluded","NavigateKitchen","DeliverCanToBowl")
    all_tasks=primary_tasks+("DeliverBrewedCoffee",)
    if task not in ('all',*all_tasks):raise typer.BadParameter("unknown task")
    if strategy not in ("all","fixed","random","active"):raise typer.BadParameter("unknown strategy")
    if backend not in ('all','tabletop','robocasa'):raise typer.BadParameter('unknown backend')
    allowed=all_tasks if backend=='all' else (all_tasks[:3] if backend=='tabletop' else all_tasks[3:])
    # The coffee task remains an explicit stretch target.  The default small
    # validation stays at five representative tasks / 15 total episodes.
    available=primary_tasks if backend=='all' else (primary_tasks[:3] if backend=='tabletop' else primary_tasks[3:])
    tasks=available if task=="all" else (task,)
    if any(t not in allowed for t in tasks):raise typer.BadParameter('task/backend mismatch')
    strategies=("fixed","random","active") if strategy=="all" else (strategy,)
    episodes=[f"{t}_{m}_{s:04d}" for t in tasks for m in strategies for s in settings.validation_seeds]
    if len(episodes)>15:raise typer.BadParameter('small validation is limited to 15 episodes total; choose fewer tasks, strategies or validation_seeds')
    expected={"kind":"small_development_validation","fingerprint":fingerprint(settings),
        "config":settings.model_dump(mode="json"),"tasks":list(tasks),"strategies":list(strategies),
        "episodes":episodes}
    if dry_run:
        print(json.dumps(expected,indent=2));return
    if any(t in all_tasks[3:] for t in tasks):
        from .runtime import mobile_process
        mobile_process(['doctor','--render'])
    root.mkdir(parents=True,exist_ok=True);manifest=root/"experiment.json"
    if manifest.exists():
        if json.loads(manifest.read_text())!=expected:
            raise typer.BadParameter("benchmark code/config/protocol changed; use a new root")
    else:
        if list(root.glob("*/result.json")):
            raise typer.BadParameter("existing episodes lack a frozen protocol; use a new root")
        manifest.write_text(json.dumps(expected,indent=2))
    for name in tasks:
        for method in strategies:
            for seed in settings.validation_seeds:
                result=root/f"{name}_{method}_{seed:04d}"/"result.json"
                if not result.exists():
                    if name in all_tasks[3:]:
                        from .runtime import mobile_process
                        args=['run','--task',name,'--seed',seed,'--strategy',method,'--root',root.resolve()]
                        if config:args+=['--config',config.resolve()]
                        mobile_process(args)
                    else:
                        from .runner import run_episode
                        run_episode(name,seed,method,root,config=settings)
                else:
                    old=json.loads(result.read_text());meta=json.loads((result.parent/"manifest.json").read_text())
                    if not old["online"] or meta["source_sha256"]!=expected["fingerprint"]:
                        raise typer.BadParameter("episode is replayed or incompatible with frozen protocol")
    summarize(root)

@app.command()
def summarize(root:Path=Path("runs/validation")):
    """Report every completed episode, including failures and missing runs."""
    from collections import Counter
    from .evaluation import wilson
    from .evaluation import failure_category
    protocol=json.loads((root/"experiment.json").read_text()) if (root/"experiment.json").exists() else None
    all_rows=[(p.parent.name,json.loads(p.read_text())) for p in root.glob("*/result.json")]
    rows=[r for name,r in all_rows if r["online"] and (protocol is None or name in protocol["episodes"])]
    summary=[]
    for task in ("block","mug","occluded","NavigateKitchen","DeliverCanToBowl","DeliverBrewedCoffee"):
        for strategy in ("fixed","random","active"):
            group=[r for r in rows if r["task"]==task and r["strategy"]==strategy]
            n=len(group); k=sum(r["success"] for r in group)
            def mean(key):
                values=[r[key] for r in group if r.get(key) is not None]
                return sum(values)/len(values) if values else None
            summary.append({"task":task,"strategy":strategy,"n":n,"successes":k,"success_rate":k/n if n else None,
                "wilson_95":wilson(k,n),"first_grasp_success_rate":None if task=='NavigateKitchen' else mean("first_grasp_success"),
                "collision_rate":mean("collision"),"mean_observations":mean("observations"),"mean_vlm_calls":mean("vlm_calls"),
                "executed_episodes":sum(r["attempts"]>0 for r in group),"mean_grasp_attempts":mean("attempts"),
                "mean_inference_seconds":mean("inference_seconds"),"mean_planning_seconds":mean("planning_seconds"),
                "mean_perception_seconds":mean('perception_seconds'),"mean_segmentation_calls":mean('segmentation_calls'),
                "grounding_types":dict(Counter(kind for r in group for kind in (r.get('grounding_types') or []))),
                "official_success_rate":mean('official_success'),"collision_free_success_rate":mean('collision_free_success'),
                "mean_path_length_m":mean('path_length_m'),"mean_held_path_length_m":mean('held_path_length_m'),
                "geometry_rejections":dict(Counter(reason for r in group for reason in r.get('geometry_rejections',[]))),
                "failure_categories":dict(Counter(failure_category(r["failure"]) for r in group if not r["success"])),
                "failures":dict(Counter(r["failure"] for r in group if not r["success"]))})
    root.mkdir(parents=True,exist_ok=True)
    completed_names={name for name,r in all_rows if r["online"]}
    (root/"summary.json").write_text(json.dumps({"kind":protocol.get('kind','benchmark') if protocol else "development",
        "completed":len(rows),"expected":len(protocol["episodes"]) if protocol else None,
        "missing":sorted(set(protocol["episodes"])-completed_names) if protocol else [],
        "replayed_excluded":sum(not r["online"] for _,r in all_rows),"groups":summary},indent=2))
    print(json.dumps(summary,indent=2))

@app.command()
def replay(episode:Path,output:Path|None=None):
    """Re-render recorded simulation states offline; never counted as an online trial."""
    record=json.loads((episode/'result.json').read_text())
    if record['task'] in ('NavigateKitchen','DeliverCanToBowl','DeliverBrewedCoffee'):
        from .runtime import mobile_process
        mobile_process(['replay','--episode',episode.resolve(),'--output',(output or episode/'replay.mp4').resolve()]);return
    from .video import replay_episode
    replay_episode(episode,output or episode/"replay.mp4")

@app.command("render-video")
def render_video(episodes:list[Path],output:Path=Path("demo.mp4"),annotations:Path|None=None):
    """Compose a bilingual 1080p presentation from actual episode recordings."""
    from .video import compose_demo
    compose_demo(episodes,output,annotations)

if __name__=="__main__": app()
