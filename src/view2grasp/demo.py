"""Captioned development demo built exclusively from recorded image evidence.

Static observation/geometry panels are editorial inserts, never synchronized
sensor claims. Source videos preserve every frame and their simulation speed.
"""
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .video import font

W, H, FPS = 1920, 1080, 30
BG = (14, 20, 27)
PANEL = (23, 32, 42)
WHITE = (231, 237, 243)
MUTED = (154, 174, 192)
TEAL = (97, 220, 191)
AMBER = (255, 197, 103)
BLUE = (100, 177, 255)
F = lru_cache(maxsize=32)(font)


def text(im, xy, content, size=28, color=WHITE):
    ImageDraw.Draw(im).text(xy, content, font=F(size), fill=color)


def fit(im, box, picture):
    x, y, width, height = box
    source = ImageOps.contain(picture.convert('RGB'), (width, height), Image.Resampling.LANCZOS)
    im.paste(source, (x + (width-source.width)//2, y + (height-source.height)//2))


def wrap(im, value, xy, width, size=28, color=WHITE, gap=12):
    x, y = xy
    line = ''
    for char in value:
        if char == '\n' or F(size).getlength(line+char) > width:
            text(im, (x, y), line, size, color)
            y += size+gap
            line = '' if char == '\n' else char
        else:
            line += char
    if line:
        text(im, (x, y), line, size, color)
    return y+size+gap


def canvas(title, subtitle, badge='DEVELOPMENT DEMO  /  V1'):
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    d.line((64, 51, 112, 51), fill=TEAL, width=5)
    text(im, (130, 33), 'VIEW2GRASP', 27, TEAL)
    text(im, (1080, 35), badge, 23, AMBER)
    text(im, (64, 99), title, 48)
    text(im, (66, 167), subtitle, 27, MUTED)
    d.line((64, 218, 1856, 218), fill=(50, 66, 81), width=2)
    return im


def captions(im, cn, en):
    ImageDraw.Draw(im).rectangle((0, 927, W, H), fill=BG)
    text(im, (64, 948), cn, 31)
    text(im, (64, 1001), en, 25, MUTED)


def source_mode(result, annotation):
    mode = 'ONLINE VLM' if result['online'] else 'VLM RESPONSE REPLAY'
    return ('EARLIER BUILD / ' if annotation.get('historical') else '') + mode


def grounding_frame(episode):
    candidates = sorted(episode.glob('candidates_*.npz'))
    if candidates:
        return int(candidates[0].stem.split('_')[-1])
    return 1


def roi(episode, frame_id):
    """Use recorded semantic extents only, preserving the image aspect ratio."""
    path = episode/'observations'/f'decision_{frame_id:03d}.json'
    d = json.loads(path.read_text()) if path.exists() else {}
    boxes = [g['box'] for g in d.get('groundings', []) if g.get('box')]
    if d.get('bbox'):
        boxes.append(d['bbox'])
    if not boxes:
        return (0, 0, 512, 512)
    b = np.asarray(boxes)
    x0, y0 = b[:, :2].min(0)*511
    x1, y1 = b[:, 2:].max(0)*511
    cx, cy = (x0+x1)/2, (y0+y1)/2
    side = min(512, max(96, x1-x0+45, y1-y0+45))
    left, top = np.clip([cx-side/2, cy-side/2], 0, 512-side)
    return tuple(map(int, (left, top, left+side, top+side)))


def observation_panel(episode, frame_id, result, annotation):
    path = episode/'observations'/f'frame_{frame_id:03d}.png'
    grounded = path.with_name(f'grounding_{frame_id:03d}.png')
    meta = np.load(path.with_suffix('.npz'))
    az, el, distance = meta['view']
    im = canvas(annotation['title'], f'{episode.name}  /  OBSERVATION {frame_id:03d}', source_mode(result, annotation))
    fit(im, (64, 238, 680, 680), Image.open(grounded if grounded.exists() else path))
    text(im, (792, 257), '主动观察 / Camera view', 34, TEAL)
    text(im, (792, 320), f'Azimuth {az:.0f}°   Elevation {el:.0f}°   Range {distance:.2f} m', 28)
    text(im, (792, 375), f'Camera: {str(meta["camera_id"]) if "camera_id" in meta else "activeview"}', 25, MUTED)
    if grounded.exists():
        fit(im, (792, 441, 470, 445), Image.open(grounded).crop(roi(episode, frame_id)))
        wrap(im, '语义提示放大\nSemantic crop\n\n绿：目标框\n青：抓取部位\n黄：接触点\n紫：放置区域', (1300, 460), 530, 27)
    else:
        wrap(im, '相机实际移动后采集新图像。\n多视角深度用于补充可见空间。', (792, 490), 930, 35)
        wrap(im, 'New images are acquired after camera motion. Depth observations update scene coverage.', (792, 648), 910, 29, MUTED)
    captions(im, '观测回顾：每帧停留 1.2 秒，推理与规划等待已压缩。', 'Observation review: 1.2 seconds per view; inference and planning waits condensed.')
    return im


def geometry_panel(episode, frame_id, result, annotation):
    im = canvas('像素 → 三维表面 → 抓取候选', 'Recorded grounding and geometric candidates', source_mode(result, annotation))
    raw = Image.open(episode/'observations'/f'frame_{frame_id:03d}.png').convert('RGB')
    geometry = episode/f'grounding_{frame_id:03d}_geometry.png'
    semantic = episode/'observations'/f'grounding_{frame_id:03d}.png'
    fit(im, (64, 293, 540, 540), Image.open(semantic if semantic.exists() else episode/'observations'/f'frame_{frame_id:03d}.png').crop(roi(episode, frame_id)))
    fit(im, (642, 293, 540, 540), Image.open(geometry if geometry.exists() else semantic).crop(roi(episode, frame_id)))
    candidates = np.load(episode/f'candidates_{frame_id:03d}.npz')
    meta = np.load(episode/'observations'/f'frame_{frame_id:03d}.npz')
    points = candidates['positions']
    camera = (points-meta['T_wc'][:3, 3]) @ meta['T_wc'][:3, :3]
    projected = camera @ meta['K'].T
    uv = projected[:, :2]/np.maximum(projected[:, 2:], 1e-8)
    d = ImageDraw.Draw(raw)
    for (x, y), z in zip(uv[::max(1, len(uv)//90)], camera[::max(1, len(uv)//90), 2]):
        if z > 0 and 0 <= x < raw.width and 0 <= y < raw.height:
            d.ellipse((x-1.5, y-1.5, x+1.5, y+1.5), fill=AMBER)
    fit(im, (1220, 293, 636, 540), raw.crop(roi(episode, frame_id)))
    text(im, (64, 245), '01  VLM 语义提议', 27, TEAL)
    text(im, (642, 245), '02  深度验证后的区域' if geometry.exists() else '02  旧版区域提示', 27, TEAL)
    text(im, (1220, 245), f'03  {len(points)} 个候选 / candidates', 27, TEAL)
    text(im, (64, 859), f'Frame {frame_id:03d}  |  Same-image crops  |  Candidate projections are proposals, not executed paths', 25, MUTED)
    captions(im, '区域与深度决定表面；围绕相机射线搜索 15° 内的接近方向。', 'Regions and depth define surfaces; candidate approaches stay within a 15° ray cone.')
    return im


def srt_time(seconds):
    ms = round(seconds*1000)
    return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'


def compose(episodes, output, annotations=None):
    episodes = [Path(p) for p in episodes]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    notes = json.loads(Path(annotations).read_text()) if annotations else {}
    segments, sources, subtitles = [], [], []
    previews = output.parent/(output.stem+'_previews')
    previews.mkdir(exist_ok=True)
    frame_count = 0
    with imageio.get_writer(output, fps=FPS, codec='libx264', quality=8,
                            macro_block_size=1, ffmpeg_params=['-preset', 'fast', '-movflags', '+faststart']) as writer:
        def hold(im, seconds, label, cn, en):
            nonlocal frame_count
            start = frame_count/FPS
            captions(im, cn, en)
            im.save(previews/f'{len(segments):02d}_{label}.jpg', quality=90)
            rgb = np.asarray(im)
            count = round(seconds*FPS)
            for _ in range(count):
                writer.append_data(rgb)
            frame_count += count
            segments.append(dict(label=label, start=start, end=frame_count/FPS))
            subtitles.append((start, frame_count/FPS, cn, en))

        def animate(draw_frame, seconds, label, cn, en):
            nonlocal frame_count
            start = frame_count/FPS
            # Diagnostic animation updates at 10 Hz; encode on the 30 fps canvas.
            count = round(seconds*10)
            for i in range(count):
                im = draw_frame(i/max(1, count-1))
                captions(im, cn, en)
                if i in (0, count//2, count-1):
                    im.save(previews/f'{len(segments):02d}_{label}_{i:03d}.jpg', quality=90)
                rgb = np.asarray(im)
                for _ in range(3):
                    writer.append_data(rgb)
                    frame_count += 1
            segments.append(dict(label=label, start=start, end=frame_count/FPS, type='explanatory_animation'))
            subtitles.append((start, frame_count/FPS, cn, en))

        im = canvas('让视觉提示驱动机器人行动', 'Multi-form grounding for observation, grasping and placement')
        text(im, (70, 300), 'VIEW', 108, TEAL)
        text(im, (420, 300), '→', 108, MUTED)
        text(im, (600, 300), 'GROUND', 108)
        text(im, (1210, 300), '→', 108, MUTED)
        text(im, (1390, 300), 'ACT', 108, AMBER)
        text(im, (73, 524), 'Franka Panda  /  robosuite  /  RGB-D  /  Local VLM', 39)
        wrap(im, '三个桌面任务 · 实际仿真记录 · 中英字幕', (73, 664), 1700, 38)
        text(im, (73, 763), 'Development snapshot / selected runs across versions / no benchmark claim', 29, MUTED)
        hold(im, 9, 'intro', '第一版开发演示：保留成功、回放与尚未完成的阶段。', 'First development demo: successes, response replay and unfinished stages are labeled.')

        im = canvas('从任务语言到可执行轨迹', 'A visual proposal becomes an action only after geometric checks')
        steps = [('01', '观察与选视角', 'Observe / choose a view'), ('02', '点、框与区域', 'Points / boxes / regions'), ('03', '深度与候选', 'Depth / grasp candidates'), ('04', '规划与执行', 'Plan / execute / verify')]
        for i, (n, cn, en) in enumerate(steps):
            x = 64+i*453
            ImageDraw.Draw(im).rectangle((x, 310, x+425, 674), fill=PANEL)
            text(im, (x+25, 340), n, 67, TEAL)
            text(im, (x+25, 456), cn, 33)
            wrap(im, en, (x+25, 530), 375, 26, MUTED)
        text(im, (64, 748), 'RGB-D → 5 mm map → robot-only IK + RRT-Connect → joint control', 34)
        hold(im, 12, 'pipeline', 'VLM 提议语义目标；几何规划器负责可达性和路径检查。', 'The VLM proposes semantic targets; geometry checks reachability and motion paths.')

        from .demo_principles import add_principles
        add_principles(episodes, hold, animate)

        for episode in episodes:
            r = json.loads((episode/'result.json').read_text())
            manifest = json.loads((episode/'manifest.json').read_text())
            a = dict(title={'block': '方块抓起', 'mug': '杯柄抓取', 'occluded': '遮挡取放'}.get(r['task'], r['task']), subtitle=r['task'], cn='实际运行记录。', en='Recorded episode.', historical=False)
            a.update(notes.get(episode.parent.name, {}))
            mode = source_mode(r, a)
            status = 'SUCCESS (RECORDED)' if r['success'] else 'PARTIAL / LIFT BLOCKED' if r['task']=='mug' else 'NOT SUCCESSFUL'
            sources.append(dict(episode=str(episode), result=r, annotation=a, source_mode=mode,
                                source_sha256=manifest.get('source_sha256'),
                                execution_sha256=sha256((episode/'execution.mp4').read_bytes()).hexdigest()))
            print(f'Rendering {episode}: {mode}, {status}', flush=True)
            im = canvas(a['title'], a['subtitle'], mode)
            text(im, (64, 295), status, 49, TEAL if r['success'] else AMBER)
            text(im, (64, 398), f"{r['observations']} views     {r['attempts']} grasp attempt     {r['vlm_calls']} VLM calls", 35)
            text(im, (64, 485), f"Inference {r['inference_seconds']:.1f}s    Planning {r['planning_seconds']:.1f}s", 30, MUTED)
            wrap(im, str(episode), (64, 594), 1750, 27, MUTED)
            text(im, (64, 749), f"Source build: {manifest.get('source_sha256', 'unavailable')[:16]}", 28, MUTED)
            hold(im, 6, r['task']+'_chapter', a['cn'], a['en'])

            for path in sorted((episode/'observations').glob('frame_*.png')):
                idx = int(path.stem.split('_')[-1])
                im = observation_panel(episode, idx, r, a)
                hold(im, 1.2, r['task']+f'_view_{idx}', '观测回顾：每帧 1.2 秒；实际执行将在下一段展示。', 'Observation review: 1.2 seconds per view; recorded execution follows separately.')
            fid = grounding_frame(episode)
            hold(geometry_panel(episode, fid, r, a), 8, r['task']+'_geometry',
                 '左：语义提议；中：区域验证；右：三维候选在相机图像上的投影。',
                 'Left: semantic proposal. Middle: region evidence. Right: projected 3D candidates.')

            reader = imageio.get_reader(episode/'execution.mp4')
            source_fps = reader.get_meta_data()['fps']
            start = frame_count/FPS
            base = canvas(a['title'], f'{episode.name}  /  RECORDED EXECUTION', mode)
            ref = episode/'observations'/f'grounding_{fid:03d}.png'
            if not ref.exists():
                ref = ref.with_name(f'frame_{fid:03d}.png')
            fit(base, (1300, 287, 540, 390), Image.open(ref).crop(roi(episode, fid)))
            text(base, (1300, 239), 'Grounding reference', 29, TEAL)
            wrap(base, f'Static reference / frame {fid:03d}\nNot a live camera feed\n\n640 × 480 source footage\n1080p presentation canvas\n1× simulation playback', (1300, 712), 530, 25, MUTED, 9)
            cn = '实际执行 · 原始速度；右侧为抓取前定位参考图。'
            en = 'Actual execution at 1× simulation time; right panel is a static grounding reference.'
            captions(base, cn, en)
            out_index = 0
            for source_index, rgb in enumerate(reader):
                im = base.copy()
                fit(im, (64, 238, 1180, 675), Image.fromarray(rgb))
                text(im, (1040, 865), f'{source_index/source_fps:04.1f}s', 25, TEAL)
                if source_index == 0:
                    im.save(previews/f'{len(segments):02d}_{r["task"]}_execution.jpg', quality=90)
                end_index = round((source_index+1)*FPS/source_fps)
                for _ in range(end_index-out_index):
                    writer.append_data(np.asarray(im))
                    frame_count += 1
                out_index = end_index
            reader.close()
            segments.append(dict(label=r['task']+'_execution', start=start, end=frame_count/FPS, source=str(episode/'execution.mp4'), speed=1))
            subtitles.append((start, frame_count/FPS, cn, en))
            im = canvas('这一段证明了什么', 'Recorded outcome and its scope', mode)
            text(im, (64, 326), status, 60, TEAL if r['success'] else AMBER)
            text(im, (64, 446), f"Evaluator success: {r['success']}   /   Recorded collision: {r['collision']}", 34)
            wrap(im, a['cn'], (64, 570), 1720, 39)
            wrap(im, r['failure'] or ('Earlier-build result; current-build validation pending.' if a.get('historical') else 'Selected development episode; not an aggregate benchmark.'), (64, 716), 1720, 31, MUTED)
            hold(im, 7, r['task']+'_outcome', a['cn'], a['en'])

        im = canvas('当前进度与下一步', 'A working prototype, with evidence and remaining work')
        rows = [('方块抓起', '新版在线成功 / current online success'), ('遮挡取放', '新版响应回放成功 / response replay success'), ('杯柄抓取', '旧版在线成功；新版抬升待完成 / earlier success; updated lift pending')]
        for i, (cn, en) in enumerate(rows):
            y = 299+i*145
            text(im, (64, y), cn, 38, TEAL)
            wrap(im, en, (390, y+6), 1450, 29)
        text(im, (64, 781), 'Next: current-build tabletop validation → RoboCasa navigation + simple mobile manipulation', 27, MUTED)
        hold(im, 12, 'outro', '本片是阶段演示；三种观察策略的性能对比尚未完成。', 'This is a development demo; comparative evaluation of observation policies is not complete.')

    output.with_suffix('.json').write_text(json.dumps(dict(kind='development_demo', canvas=[W, H], fps=FPS,
        duration_seconds=frame_count/FPS, audio=False, sources=sources, segments=segments,
        note='Bilingual burned-in captions. Static observation review is separate from execution. Source footage is upscaled, not native 1080p. Historical and response-replay records are labeled.'), indent=2, ensure_ascii=False))
    output.with_suffix('.srt').write_text('\n\n'.join(f'{i}\n{srt_time(s)} --> {srt_time(e)}\n{cn}\n{en}' for i, (s,e,cn,en) in enumerate(subtitles, 1))+'\n')
    print(f'Wrote {output} ({frame_count/FPS:.1f}s)', flush=True)
