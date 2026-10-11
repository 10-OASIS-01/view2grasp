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
            'resampling': 'nearest_neighbor_endpoint_aligned_integer_centers', 'scale': self.scale,
            'content_bounds_xyxy': [x0, y0, x0 + (w - 1) * self.scale + 1,
                y0 + (h - 1) * self.scale + 1],
            'bounds_convention': 'left_top_inclusive_right_bottom_exclusive',
            'source_to_diagnostic_pixel_centers': {'scale_xy': [self.scale, self.scale],
                'offset_xy': [x0, y0]},
            'normalized_pixel_centers_match_primary': True,
            'external_margins_or_legend': False,
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
    # Align the first/last pixel CENTERS, so normalized coordinates are exactly
    # shared with the primary image: scale*x/(scale*(w-1)) == x/(w-1).
    # Ordinary repeat/resize adds half-pixel margins and would not satisfy this
    # identity. No header, axes, padding, or legend creates a second image frame.
    out_h, out_w = (h - 1) * scale + 1, (w - 1) * scale + 1
    rows = np.rint(np.arange(out_h) / scale).astype(int)
    columns = np.rint(np.arange(out_w) / scale).astype(int)
    canvas = Image.fromarray(rgb[rows[:, None], columns[None, :]])
    draw = ImageDraw.Draw(canvas)
    # Lines are between source centers, including odd scale factors. At image
    # edges draw only existing interior borders; clamping an outside border to
    # the image edge would overwrite the first/last source pixel center.
    half = scale // 2
    for x in range(grid_step, w, grid_step):
        xx = x * scale - half
        draw.line((xx, 0, xx, out_h - 1), fill=(225, 225, 225), width=1)
    for y in range(grid_step, h, grid_step):
        yy = y * scale - half
        draw.line((0, yy, out_w - 1, yy), fill=(225, 225, 225), width=1)
    for x, y in rejected:
        left, right = x * scale - half, x * scale + (scale + 1) // 2
        top, bottom = y * scale - half, y * scale + (scale + 1) // 2
        for xx in (left, right):
            if 0 <= xx < out_w:
                draw.line((xx, max(0, top), xx, min(out_h - 1, bottom)), fill=(220, 0, 160), width=1)
        for yy in (top, bottom):
            if 0 <= yy < out_h:
                draw.line((max(0, left), yy, min(out_w - 1, right), yy), fill=(220, 0, 160), width=1)
    result = np.asarray(canvas).copy();result.setflags(write=False)
    return AnnotationDiagnostic(result, int(obs.frame_id), obs.camera_id, (h, w), _rgb_hash(rgb),
        scale, (0, 0), grid_step, tuple(rejected))
