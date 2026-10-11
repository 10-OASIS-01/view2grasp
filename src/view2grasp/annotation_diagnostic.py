"""Pixel-location aids derived only from the current annotation image.

The unmarked image remains the sole semantic evidence and output coordinate
system. This companion image never supplies a region or contact permission.
"""
from dataclasses import dataclass
import hashlib

import numpy as np
from PIL import Image, ImageDraw


def _rgb_hash(rgb):
    return hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()


@dataclass(frozen=True)
class AnnotationDiagnostic:
    rgb: np.ndarray
    source_frame_id: int
    source_camera_id: str
    source_shape_hw: tuple[int, int]
    source_rgb_sha256: str
    scale: int
    content_offset_xy: tuple[int, int]
    grid_step: int
    rejected_pixels: tuple[tuple[int, int], ...]

    def validate_source(self, obs):
        if (obs.frame_id != self.source_frame_id or obs.camera_id != self.source_camera_id
                or tuple(obs.rgb.shape[:2]) != self.source_shape_hw
                or _rgb_hash(obs.rgb) != self.source_rgb_sha256):
            raise ValueError('annotation diagnostic does not match the current unmarked image')

    def provenance(self):
        h, w = self.source_shape_hw
        x0, y0 = self.content_offset_xy
        return {'kind': 'pixel_coordinate_diagnostic_of_current_image',
            'source_frame_id': self.source_frame_id, 'source_camera_id': self.source_camera_id,
            'source_shape_hw': [h, w], 'source_rgb_sha256': self.source_rgb_sha256,
            'diagnostic_shape_hw': list(self.rgb.shape[:2]), 'diagnostic_rgb_sha256': _rgb_hash(self.rgb),
            'resampling': 'nearest_neighbor_integer_scale', 'scale': self.scale,
            'content_bounds_xyxy': [x0, y0, x0 + w * self.scale, y0 + h * self.scale],
            'bounds_convention': 'left_top_inclusive_right_bottom_exclusive',
            'grid_step_source_pixels': self.grid_step,
            'rejected_pixels': [{'id': f'R{i + 1}', 'pixel': [x, y],
                'normalized': [x / (w - 1), y / (h - 1)]}
                for i, (x, y) in enumerate(self.rejected_pixels)],
            'return_coordinates': 'FIRST unmarked image normalized pixel centers; x/(width-1), y/(height-1)',
            'scope': 'Grid and magenta pixel borders are geometry diagnostics, NOT object labels, suggested masks, or instructions to delete pixels. Reinspect the unmarked image; omit uncertain regions.',
            'source_pixel_centers_unchanged': True, 'new_observation': False}


def build_annotation_diagnostic(obs, feedback, *, scale=4, grid_step=8):
    """Magnify the acquired image and mark rejected pixel borders, never centers.

    ``feedback`` must already use this supplied image's coordinates. Source
    full-image diagnostics must be converted by the crop caller beforehand.
    No mask is constructed, refined, accepted or modified here.
    """
    rgb = np.asarray(obs.rgb)
    if (rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8
            or min(rgb.shape[:2]) < 2 or tuple(rgb.shape[:2]) != tuple(obs.depth.shape)):
        raise ValueError('annotation diagnostic requires aligned uint8 RGB-D')
    if (isinstance(scale, bool) or not isinstance(scale, int) or not 4 <= scale <= 8
            or isinstance(grid_step, bool) or not isinstance(grid_step, int) or grid_step < 4):
        raise ValueError('annotation diagnostic requires integer scale 4..8 and grid step >=4')
    if feedback.get('frame_id') != obs.frame_id or feedback.get('camera_id') != obs.camera_id:
        raise ValueError('annotation diagnostic feedback is stale or from another camera')
    space = feedback.get('coordinate_space')
    if space == 'supplied_crop_normalized_pixel_centers':
        shape = feedback.get('image_shape_hw')
    elif space == 'source_image_normalized_pixel_centers':
        shape = feedback.get('source_shape_hw')
    else:
        raise ValueError('annotation diagnostic feedback needs an explicit image coordinate space')
    h, w = rgb.shape[:2]
    if shape is None or tuple(shape) != (h, w):
        raise ValueError('annotation diagnostic feedback shape differs from the supplied image')
    samples = feedback.get('rejected_samples', [])
    if not isinstance(samples, list) or len(samples) > 12:
        raise ValueError('annotation diagnostic accepts at most twelve rejected pixel samples')
    rejected = []
    for sample in samples:
        xy = np.asarray(sample.get('normalized'), dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or np.any((xy < 0) | (xy > 1)):
            raise ValueError('annotation diagnostic rejected pixel is outside the supplied image')
        pixel = tuple(map(int, np.rint(xy * [w - 1, h - 1])))
        if pixel not in rejected: rejected.append(pixel)
    x0 = y0 = 44
    canvas = Image.new('RGB', (x0 + w * scale + 280,
        max(y0 + h * scale + 44, y0 + 40 + len(rejected) * 15 + 60)), (248, 248, 248))
    canvas.paste(Image.fromarray(rgb).resize((w * scale, h * scale), Image.Resampling.NEAREST), (x0, y0))
    draw = ImageDraw.Draw(canvas)
    draw.text((x0, 5), 'DIAGNOSTIC ONLY - use FIRST unmarked image coordinates', fill=(0, 0, 0))
    draw.text((x0, 19), f'Raw image: {w} x {h} pixels; display: {scale}x; axis labels are RAW pixel indices', fill=(0, 0, 0))
    # Lines lie on cell borders. Every original pixel's magnified center stays
    # byte-identical, including marked pixels and intersections of grid lines.
    for x in range(0, w, grid_step):
        xx = x0 + x * scale - 1
        draw.line((xx, y0 - 1, xx, y0 + h * scale - 1), fill=(225, 225, 225), width=1)
        draw.text((xx - 5, y0 - 16), str(x), fill=(0, 0, 0))
    for y in range(0, h, grid_step):
        yy = y0 + y * scale - 1
        draw.line((x0 - 1, yy, x0 + w * scale - 1, yy), fill=(225, 225, 225), width=1)
        draw.text((5, yy - 5), str(y), fill=(0, 0, 0))
    legend_x = x0 + w * scale + 12
    draw.text((legend_x, y0), 'Rejected pixel borders (magenta):', fill=(0, 0, 0))
    for i, (x, y) in enumerate(rejected):
        left, top = x0 + x * scale, y0 + y * scale
        draw.rectangle((left, top, left + scale - 1, top + scale - 1), outline=(220, 0, 160), width=1)
        draw.text((legend_x, y0 + 20 + i * 15), f'R{i + 1}: x={x}, y={y}  ({x/(w-1):.6f}, {y/(h-1):.6f})', fill=(100, 0, 80))
    legend_y = max(y0 + 20 + len(rejected) * 15, y0 + 45)
    for i, line in enumerate(('Not object identity or a mask.', 'Do not blindly erase marked pixels.',
                              'Reinspect visible material/boundaries.', 'Return FIRST-image coordinates.')):
        draw.text((legend_x, legend_y + i * 15), line, fill=(0, 0, 0))
    result = np.asarray(canvas).copy();result.setflags(write=False)
    return AnnotationDiagnostic(result, int(obs.frame_id), obs.camera_id, (h, w), _rgb_hash(rgb),
        scale, (x0, y0), grid_step, tuple(rejected))
