"""Evidence-backed method animations; schematic panels are explicitly labeled."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .demo import canvas, captions, fit, text, wrap, roi, TEAL, AMBER, BLUE, MUTED, PANEL
from .geometry import unproject


def projector(points, box):
    """Fixed orthographic diagnostic view, not a sensor image."""
    basis = np.array([[.82, -.57, 0], [-.31, -.45, .84]])
    xy = np.asarray(points) @ basis.T
    lo, hi = xy.min(0), xy.max(0)
    x, y, w, h = box
    scale = min(w/max(hi[0]-lo[0], .01), h/max(hi[1]-lo[1], .01))*.85
    center = (hi+lo)/2

    def project(p):
        a = (np.asarray(p) @ basis.T-center)*scale
        return a*np.array([1, -1])+np.array([x+w/2, y+h/2])
    return project


def line3(im, project, points, color, width=3):
    ImageDraw.Draw(im).line([tuple(p) for p in project(points)], fill=color, width=width)


def dot3(im, project, point, color, radius=6, label=None):
    x, y = project(point)
    ImageDraw.Draw(im).ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
    if label:
        text(im, (x+12, y-22), label, 24, color)


def depth_cloud(episode, frame):
    meta = np.load(episode/'observations'/f'frame_{frame:03d}.npz')
    v, u = np.mgrid[0:512:5, 0:512:5]
    points = unproject(np.stack([u, v], -1), meta['depth'][::5, ::5], meta['K'], meta['T_wc']).reshape(-1, 3)
    good = np.isfinite(points).all(1) & (np.abs(points[:, 0]) < .65) & (np.abs(points[:, 1]) < .65) & (points[:, 2] > .74) & (points[:, 2] < 1.5)
    return points[good]


def recorded_robot_paths(episode):
    """Offline FK only: read robot joints, never object poses, from a log."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(episode/'scene.xml'))
    data = mujoco.MjData(model)
    addresses = np.array([model.jnt_qposadr[model.joint(f'robot0_joint{i}').id] for i in range(1, 8)])
    site = model.site('gripper0_right_grip_site').id

    def fk(qs):
        out = []
        for q in qs:
            data.qpos[addresses] = q
            mujoco.mj_kinematics(model, data)
            out.append(data.site_xpos[site].copy())
        return np.asarray(out)
    paths = [(p.stem.rsplit('_', 1)[-1], fk(np.load(p))) for p in sorted(episode.glob('path_*.npy'))]
    states = np.load(episode/'trajectory.npz')['states']
    # Flattened MuJoCo records begin with time, then qpos; only arm qpos read.
    return paths, fk(states[:, 1+addresses])


def add_principles(episodes, hold, animate):
    mug = next((p for p in reversed(episodes) if (p/'grounding_002_validation.json').exists()), None)
    transport = next((p for p in episodes if p.name.startswith('occluded')), None)
    if mug is None or transport is None:
        return
    meta = np.load(mug/'observations/frame_002.npz')
    validation = json.loads((mug/'grounding_002_validation.json').read_text())
    uv = np.rint(np.array(validation['contact'])*511).astype(int)
    depth = float(meta['depth'][uv[1], uv[0]])
    P = unproject(uv, depth, meta['K'], meta['T_wc'])
    C = meta['T_wc'][:3, 3]
    ray = (P-C)/np.linalg.norm(P-C)
    candidates = np.load(mug/'candidates_002.npz')

    im = canvas('原理 00 / 先决定从哪里看', 'The agent selects the next view before a new image is acquired', 'METHOD / ACTUAL CAMERA OBSERVATIONS')
    for x, frame in [(64, 1), (975, 2)]:
        obs = np.load(mug/'observations'/f'frame_{frame:03d}.npz')
        az, el, distance = obs['view']
        text(im, (x, 253), f'Frame {frame:03d}  /  az {az:.0f}°  /  el {el:.0f}°', 31, TEAL)
        fit(im, (x, 310, 845, 525), Image.open(mug/'observations'/f'frame_{frame:03d}.png'))
    text(im, (64, 855), 'Observe → choose a broadside view → move camera → acquire RGB-D → ground the handle', 29, MUTED)
    hold(im, 10, 'method_active_view', '先根据当前图像选择杯柄侧视角，再实际移动相机并获取新图像。', 'Choose a handle-side view from the current image, then move the camera and acquire a new image.')

    im = canvas('原理 01 / 语义定位与表面验证', 'The VLM proposes a part; RGB-D validates its physical surface', 'METHOD / RECORDED MUG FRAME 002')
    for x, filename, label in [(64, mug/'observations/grounding_002.png', 'VLM：框 + 带孔多边形 + 接触点'),
                                (975, mug/'grounding_002_geometry.png', 'SAM + RGB-D：验证后的表面')]:
        text(im, (x, 248), label, 30, TEAL)
        fit(im, (x, 303, 845, 530), Image.open(filename).crop(roi(mug, 2)))
    text(im, (64, 865), f"Recorded backend: {validation['source']}   |   Part pixels: {validation['part_pixels']}   |   Hole preserved", 26, MUTED)
    hold(im, 8, 'method_semantics', '杯柄孔洞保留为空：语义区域还要经过深度与表面检查。', 'The handle hole stays empty: semantic regions still require depth and surface validation.')

    def pixel_frame(t):
        im = canvas('原理 02 / 一个像素如何变成三维点', 'Back-project the grounded pixel using metric depth and camera calibration', 'METHOD / MEASURED RGB-D')
        raw = Image.open(mug/'observations/frame_002.png').convert('RGB')
        d = ImageDraw.Draw(raw)
        radius = 5+3*np.sin(t*np.pi*4)**2
        d.ellipse((uv[0]-radius, uv[1]-radius, uv[0]+radius, uv[1]+radius), outline=AMBER, width=3)
        d.line((uv[0]-15, uv[1], uv[0]+15, uv[1]), fill=AMBER, width=1)
        fit(im, (64, 255, 660, 660), raw)
        text(im, (784, 269), f'u = {uv[0]} px    v = {uv[1]} px    depth = {depth:.4f} m', 31, AMBER)
        formulas = [('1', 'p_camera = depth × inv(K) × [u, v, 1]', '像素和深度 → 相机坐标'),
                    ('2', 'P_world = R_wc × p_camera + C', '相机标定 → 世界坐标'),
                    ('3', 'a0 = (P_world - C) / ||P_world - C||', '相机到表面的射线 → 接近方向先验')]
        active = min(2, int(t*3))
        for i, (n, formula, cn) in enumerate(formulas):
            y = 368+i*148
            ImageDraw.Draw(im).rectangle((775, y-8, 1853, y+118), fill=PANEL if i <= active else (17, 23, 31))
            text(im, (795, y+4), formula, 31, TEAL if i <= active else MUTED)
            text(im, (795, y+60), cn, 27, MUTED)
        text(im, (784, 853), f'P = [{P[0]:+.3f}, {P[1]:+.3f}, {P[2]:+.3f}] m', 31, AMBER)
        return im
    animate(pixel_frame, 10, 'method_backprojection', '这里使用杯柄实测像素、米制深度和相机位姿进行计算。', 'Computed from the recorded handle pixel, metric depth and calibrated camera pose.')

    target = candidates['target']
    local = target[np.linalg.norm(target-P, axis=1) < .15]
    toward_camera = P-.15*ray
    project = projector(np.vstack([local, P-.19*ray, P+.06*ray, P+np.eye(3)*.09, P-np.eye(3)*.09]), (64, 270, 1020, 610))
    base = canvas('原理 03 / 射线不是完整的抓取姿态', 'Search approach direction, rotation, insertion and gripper width', 'METHOD / RECORDED CANDIDATES')
    for point in local[::3]:
        dot3(base, project, point, (73, 102, 111), 2)
    # Draw the actual cone convention: candidate z axes point toward the part.
    ref = np.array([0., 0., 1.]) if abs(ray[2]) < .95 else np.array([1., 0., 0.])
    bx = np.cross(ray, ref); bx /= np.linalg.norm(bx)
    by = np.cross(ray, bx)
    ring = np.array([P-.12*ray+.12*np.tan(np.deg2rad(15))*(np.cos(a)*bx+np.sin(a)*by) for a in np.linspace(0, 2*np.pi, 49)])
    line3(base, project, ring, (83, 113, 128), 2)
    for pt in ring[::8]:
        line3(base, project, [P, pt], (60, 83, 97), 1)
    line3(base, project, [toward_camera, P], TEAL, 4)
    dot3(base, project, toward_camera, TEAL, label='toward camera C')
    dot3(base, project, P, (255, 114, 181), label='P: surface')
    text(base, (1110, 284), '15° approach cone', 41, TEAL)
    wrap(base, '黄色：指间抓取中心 G\n蓝色：末端控制坐标 TCP\n粉色：点击到的表面 P\n\nG = P + insertion × approach\nTCP = G - R × tool_offset', (1110, 363), 730, 29)
    text(base, (64, 864), 'Observed target points / shortened camera ray / diagnostic 3D view', 26, MUTED)

    def ray_frame(t):
        im = base.copy()
        idx = min(len(candidates['positions'])-1, int(t*(len(candidates['positions'])-1)))
        R, G, tcp = candidates['rotations'][idx], candidates['grasp_centers'][idx], candidates['positions'][idx]
        width = float(candidates['widths'][idx])
        approach, closing = R[:, 2], R[:, 0]
        line3(im, project, [G-.09*approach, G], AMBER, 3)
        for sign in (-1, 1):
            finger = G+sign*closing*width/2
            line3(im, project, [finger-.045*approach, finger+.008*approach], AMBER, 5)
        line3(im, project, [G-closing*width/2-.045*approach, G+closing*width/2-.045*approach], AMBER, 5)
        dot3(im, project, G, AMBER, 6)
        dot3(im, project, tcp, BLUE, 5)
        text(im, (1110, 724), f"Candidate {idx+1}/{len(candidates['positions'])}", 32, AMBER)
        text(im, (1110, 785), f"Ray offset {candidates['ray_angles'][idx]:.1f}°  |  Width {width*1000:.0f} mm", 29)
        return im
    animate(ray_frame, 14, 'method_ray_candidates', '动画枚举日志中的候选；尚未通过规划的姿态不会直接执行。', 'Animation cycles through recorded proposals; only a feasible plan can be executed.')

    # Deliberately illustrative 2D occupancy slice; no inferred measured voxels.
    def mapping_frame(t):
        im = canvas('原理 04 / 看不见的空间，需要补充观察', 'Unknown space blocks motion; new observations provide evidence', 'METHOD / SCHEMATIC + REAL LOG')
        d = ImageDraw.Draw(im)
        origin = np.array([85, 333]); cell = 32
        phase = min(2, int(t*3))
        for y in range(14):
            for x in range(29):
                obstacle = (x in (13, 14) and 4 <= y <= 10)
                seen = x < 13 or (phase >= 1 and y < 4) or (phase >= 2 and y > 10)
                color = (226, 120, 109) if obstacle else (44, 114, 103) if seen else (47, 57, 72)
                a = origin+np.array([x, y])*cell
                d.rectangle((*a, *(a+cell-3)), fill=color)
        text(im, (84, 265), '二维示意切片 / illustrative occupancy slice', 29, MUTED)
        route = [(160, 550), (430, 550), (610, 550), (900, 550)] if phase == 0 else [(160, 550), (340, 397), (900, 397), (900, 550)]
        d.line(route, fill=AMBER, width=5)
        text(im, (85, 783), '黄色为待检查的路径提议 / yellow path is not yet certified', 22, AMBER)
        text(im, (85, 814), '绿：已观测自由    红：占据    灰：未知', 29)
        text(im, (85, 860), 'Schematic only; not a reconstruction of this episode.', 25, MUTED)
        wrap(im, '完整机器人 + 持物包络\n沿路径连续检查\n\n未知 / 占据 → 拒绝\n反馈 → 选视角 → 再观测', (1120, 283), 690, 33)
        events = json.loads((mug/'events.json').read_text())
        residual = [e['rejected_unknown_points'] for e in events if e['kind']=='postgrasp_residual']
        text(im, (1120, 630), '新版杯柄的真实日志', 31, AMBER)
        wrap(im, 'Post-grasp residual checks:\n'+' → '.join(map(str, residual))+' unknown points\nFinal result: lift blocked', (1120, 688), 700, 28, MUTED)
        return im
    animate(mapping_frame, 12, 'method_unknown_space', '语义上的“可走”不等于几何自由；新增观测后仍需重新检查。', 'Semantic traversability does not certify free space; new views still require collision checks.')

    paths, actual = recorded_robot_paths(transport)
    cloud = depth_cloud(transport, 1)
    p3 = projector(np.vstack([cloud, actual]+[p for _, p in paths]), (64, 259, 1100, 620))
    base = canvas('原理 05 / 规划轨迹如何变成真实动作', 'Recorded joint plans and measured robot states, visualized through forward kinematics', 'METHOD / ACTUAL OCCLUDED RUN')
    for point in cloud[::2]:
        dot3(base, p3, point, (57, 77, 86), 1)
    colors = [BLUE, TEAL, AMBER, (211, 154, 246), (255, 131, 155)]
    for i, (phase, xyz) in enumerate(paths):
        line3(base, p3, xyz, colors[i%len(colors)], 4)
        text(base, (1200, 294+i*59), phase, 30, colors[i%len(colors)])
    wrap(base, '线：日志中的规划路径\n亮点：实际关节状态推算的 TCP\n\n预抓取 / 搬运：关节空间规划\n接近：沿抓取轴分段 IK\n控制器：跟踪关节目标', (1200, 635), 620, 27)
    text(base, (64, 868), 'Observed RGB-D cloud + robot FK / offline 3D visualization', 27, MUTED)

    def trajectory_frame(t):
        im = base.copy()
        count = max(1, round(t*(len(actual)-1))+1)
        if count > 1:
            line3(im, p3, actual[:count], (237, 239, 233), 2)
        dot3(im, p3, actual[count-1], (255, 255, 255), 8)
        return im
    animate(trajectory_frame, 12, 'method_motion_plan', '规划线和实际轨迹分别显示；这里只读取机器人关节状态做离线可视化。', 'Planned paths and measured motion are distinct; this offline view uses robot joint states.')
