"""Official Hugging Face fallback for unavailable upstream Box downloads.

Downloads only public RoboCasa/NVIDIA repositories, verifies LFS SHA256, and
extracts beneath this project's isolated asset directory. No user data uploaded.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import subprocess
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'runtimes/robocasa'
ASSETS=RUNTIME/'vendor/robocasa/robocasa/models/assets'
DOWNLOADS=RUNTIME/'downloads'


def get_json(url):
    with urllib.request.urlopen(url,timeout=30) as response:return json.load(response)


def download(item):
    repo,revision,entry,dest=item
    name=entry['path'];expected=entry['lfs']['oid']
    archive=DOWNLOADS/(repo.replace('/','--')+'--'+name.replace('/','--'))
    archive.parent.mkdir(parents=True,exist_ok=True)
    def digest(path):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    if not archive.exists() or digest(archive)!=expected:
        temporary=archive.with_suffix('.part')
        url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}'
        subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','3',
                        '--connect-timeout','30','--output',str(temporary),url],check=True)
        if digest(temporary)!=expected:raise RuntimeError(f'SHA256 mismatch: {name}')
        temporary.replace(archive)
    return {'repository':repo,'revision':revision,'file':name,'sha256':expected,
            'archive':archive,'destination':dest,'bytes':entry['size']}


def main():
    if not ASSETS.is_dir():raise RuntimeError('install pinned RoboCasa checkout first')
    # This original pack supplies shared textures, baseline fixtures and Objaverse objects.
    packages=[]
    pins=json.loads((RUNTIME/'asset_sources.json').read_text())
    repo='robocasa/robocasa-assets'
    revision=pins[repo]
    for entry in get_json(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}'):
        if entry['path'] in ('textures.zip','generative_textures.zip','fixtures.zip','objaverse.zip'):
            dest=ASSETS/'objects' if entry['path']=='objaverse.zip' else ASSETS
            packages.append((repo,revision,entry,dest))
    # NVIDIA publishes the official Lightwheel assets as category archives.
    repo='nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF'
    revision=pins[repo]
    for folder,dest in [('fixtures_lightwheel',ASSETS/'fixtures'),('objects_lightwheel',ASSETS/'objects/lightwheel')]:
        for entry in get_json(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/{folder}'):
            if entry['path'].endswith('.zip'):packages.append((repo,revision,entry,dest))
    print(json.dumps({'packages':len(packages),'compressed_bytes':sum(e[2]['size'] for e in packages)}),flush=True)
    downloaded=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(download,packages):
            downloaded.append(result)
            print(f"Verified {result['file']} ({result['bytes']} bytes)",flush=True)
    # Ordered extraction intentionally overlays newer Lightwheel fixtures onto the
    # baseline pack, matching the official merged asset layout. Never extract outside it.
    for result in downloaded:
        dest=result['destination'];dest.mkdir(parents=True,exist_ok=True)
        with ZipFile(result['archive']) as archive:
            for info in archive.infolist():
                target=(dest/info.filename).resolve()
                if not target.is_relative_to(dest.resolve()) or ((info.external_attr>>16)&0o170000)==0o120000:
                    raise RuntimeError('unsafe archive member')
            archive.extractall(dest)
        print(f"Extracted {result['file']}",flush=True)
    clean=[{k:str(v.relative_to(RUNTIME.parent.parent)) if isinstance(v,Path) else v for k,v in r.items()} for r in downloaded]
    (RUNTIME/'asset_manifest.json').write_text(json.dumps({'license':'CC-BY-4.0; retain per-asset notices',
        'sources':clean},indent=2))

if __name__=='__main__':main()
