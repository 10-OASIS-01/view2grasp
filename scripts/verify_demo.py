"""Verify encoded demo, provenance, timing and representative decoded frames."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=Path)
    args = parser.parse_args()
    video = args.video
    manifest = json.loads(video.with_suffix('.json').read_text())
    reader = imageio.get_reader(video)
    metadata = reader.get_meta_data()
    assert metadata['size'] == (1920, 1080)
    assert metadata['fps'] == manifest['fps'] == 30
    assert abs(metadata['duration']-manifest['duration_seconds']) < 1/30
    assert len(manifest['sources']) == 4
    expected_start = 0
    for segment in manifest['segments']:
        assert abs(segment['start']-expected_start) < 1e-6
        assert segment['end'] > segment['start']
        expected_start = segment['end']
        if 'source' in segment:
            source = imageio.get_reader(segment['source'])
            length = source.count_frames()/source.get_meta_data()['fps']
            source.close()
            assert abs(segment['end']-segment['start']-length) < 1/30
            assert segment['speed'] == 1
    assert abs(expected_start-manifest['duration_seconds']) < 1e-6
    for source in manifest['sources']:
        episode = Path(source['episode'])
        assert source['result'] == json.loads((episode/'result.json').read_text())
        assert source['execution_sha256'] == sha256((episode/'execution.mp4').read_bytes()).hexdigest()
        if not source['result']['online']:
            assert 'REPLAY' in source['source_mode']
        if source['annotation'].get('historical'):
            assert 'EARLIER BUILD' in source['source_mode']
    samples = video.parent/(video.stem+'_decoded_checks')
    samples.mkdir(exist_ok=True)
    for i, segment in enumerate(manifest['segments']):
        if segment['label'].startswith('method_') or 'source' in segment or segment['label']=='outro':
            timestamp = (segment['start']+segment['end'])/2
            frame = reader.get_data(round(timestamp*30))
            Image.fromarray(frame).save(samples/f'{i:02d}_{segment["label"]}.jpg', quality=90)
    frame_total = reader.count_frames()
    reader.close()
    assert frame_total == round(manifest['duration_seconds']*30)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(video), '-f', 'null', '-'], check=True)
    report = dict(video=str(video), size=metadata['size'], fps=metadata['fps'],
                  duration_seconds=manifest['duration_seconds'], decoded_frames=frame_total,
                  complete_decode='passed', source_hashes='passed', result_labels='passed',
                  execution_speed='1x; all source frames retained', timeline='contiguous',
                  method_sections=sum(s['label'].startswith('method_') for s in manifest['segments']))
    video.with_suffix('.qa.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
