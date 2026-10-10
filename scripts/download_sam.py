"""Download the explicitly pinned optional SAM checkpoint and verify SHA256."""
from pathlib import Path
import hashlib
import json
import urllib.request


def main():
    root=Path(__file__).resolve().parents[1]
    spec=json.loads((root/'configs/model_weights.json').read_text())['sam2.1_hiera_tiny']
    output=root/'checkpoints/sam2.1_hiera_tiny.pt';output.parent.mkdir(exist_ok=True)
    def sha(path):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    if output.exists():
        if sha(output)!=spec['sha256']:raise RuntimeError('existing checkpoint differs; refusing to overwrite')
    else:
        temporary=output.with_suffix('.part')
        urllib.request.urlretrieve(spec['url'],temporary)
        if sha(temporary)!=spec['sha256']:raise RuntimeError('checkpoint checksum mismatch')
        temporary.replace(output)
    print(json.dumps({'checkpoint':str(output),'sha256':spec['sha256'],'verified':True}))


if __name__=='__main__':main()
