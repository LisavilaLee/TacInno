"""Tactile observation of a vision-based tactile sensor (GelSight-type gel with a 7 x 9 marker grid).

The simulator (UniVTAC) gives, per gel and frame, a depth image (240 x 320, mm from the camera) and the pixel positions
of the 63 markers, ``[reference, current]``. TacInno turns them into a 3-channel tactile observation on a 96 x 128 grid:

* indentation field ``w = clip(28.5 - depth, 0, 6) / 3`` (area-resampled 240 x 320 -> 96 x 128, 2.5 px per cell);
* marker tangential displacement field ``(u_x, u_y) / sigma_u``: displacement of each marker from the rest grid,
  missing markers in-painted on the 7 x 9 grid, then interpolated bicubically at the 96 x 128 pixel centres.

The right gel is mirrored along the image x axis so that both gels share one canonical frame. Two entry points:

* :func:`process_gel` processes a whole recorded episode of one gel (data export);
* :func:`deploy_frame` processes a single observation at deployment with the statistics of the training data;
  :func:`build_fields` then assembles the (3, 96, 128) field on the GPU, identically for training and deployment.

Also computed: a 14-d contact-state summary ``m`` (affine decomposition of the marker field, contact area, contact
centroid, maximum indentation) and the contact label ``e_c = [min depth < 28.3]``; they are the targets of the
contact-state prediction head of the policy.

Marker tracking anomalies (a marker whose reference position moved by > 1 px, whose displacement exceeds 80 px, or
that left the image) are treated like missing markers: in-painted before interpolation and ignored by the affine fit;
frames with < 50 % valid markers get ``m_valid = False``.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "W_IMG", "H_IMG", "OUT_H", "OUT_W", "N_MK", "GRID_R", "GRID_C", "DEPTH_REF", "PRESS_MAX", "PRESS_SCALE",
    "CONTACT_THR", "NO_CONTACT", "REF_TOL", "MAX_DISP", "M_NAMES", "M_CLIP",
    "canonicalize", "press_fullres", "press_field", "rest_grid", "marker_validity", "grid_order", "affine_summary",
    "depth_summary", "sanitize_m", "normalize_m", "physics_summary", "process_gel", "deploy_frame",
    "grid_sampling_coords", "inpaint_grid", "build_fields", "load_tactile_stats",
]

# Sensor image and output grid.
W_IMG, H_IMG = 320, 240
OUT_H, OUT_W = 96, 128
N_MK = 63                 # markers used (the simulator reports 64; the 64th duplicates the 63rd)
GRID_R, GRID_C = 7, 9     # marker raster (rows along y, columns along x)
# Indentation field: w = clip(DEPTH_REF - depth, 0, PRESS_MAX) / PRESS_SCALE.
DEPTH_REF, PRESS_MAX, PRESS_SCALE = 28.5, 6.0, 3.0
CONTACT_THR = 28.3        # contact where depth < 28.3 mm
NO_CONTACT = 34.0         # depth reported by the simulator without contact (camera far plane)
REF_TOL, MAX_DISP = 1.0, 80.0
M_NAMES = ['b_x', 'b_y', 'torsion', 'dilation', 'strain_1', 'strain_2', 'resid_rms', 'mean_abs_d', 'frac_d_gt1',
           'min_depth', 'contact_area', 'centroid_x', 'centroid_y', 'max_press']
M_CLIP = 30.0


def load_tactile_stats(path: str) -> Dict[str, Any]:
    """Load the tactile statistics file (rest grid, marker order, sigma_u, m / proprio normalisation)."""
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------------------------------
# NumPy preprocessing (export and deployment)
# ----------------------------------------------------------------------------------------------------------------
def canonicalize(depth, marker, gel):
    """depth (..., 240, 320); marker (..., 2, 64|63, 2) [reference, current] x points x (x, y).
    Returns float32 copies; gel 1 (right) is mirrored along x."""
    depth = np.asarray(depth, np.float32)
    mk = np.asarray(marker, np.float32)[..., :N_MK, :].copy()
    if gel == 1:
        depth = depth[..., ::-1].copy()
        mk[..., 0] = W_IMG - mk[..., 0]
    return depth, mk


def press_fullres(depth):
    """Indentation at full resolution: clip(28.5 - depth, 0, 6) / 3."""
    return np.clip(DEPTH_REF - depth, 0.0, PRESS_MAX) / PRESS_SCALE


def press_field(depth):
    """(T, 240, 320) -> (T, 96, 128) float32 indentation field (area resampling)."""
    import cv2

    w = press_fullres(depth)
    return np.stack([cv2.resize(f, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA) for f in w]).astype(np.float32)


def rest_grid(mk):
    """Episode rest grid = per-marker median of the reference positions. mk (T, 2, 63, 2) -> (63, 2)."""
    return np.median(mk[:, 0], axis=0)


def marker_validity(mk, rest):
    """mk (T, 2, 63, 2), rest (63, 2) -> valid (T, 63) bool, d (T, 63, 2) displacement current - rest."""
    ref_ok = np.linalg.norm(mk[:, 0] - rest[None], axis=-1) < REF_TOL
    d = mk[:, 1] - rest[None]
    disp_ok = np.linalg.norm(d, axis=-1) < MAX_DISP
    c = mk[:, 1]
    inside = (c[..., 0] > 0.5) & (c[..., 0] < W_IMG - 0.5) & (c[..., 1] > 0.5) & (c[..., 1] < H_IMG - 0.5)
    return ref_ok & disp_ok & inside, d.astype(np.float32)


def grid_order(rest):
    """Permutation that sorts the 63 rest positions into a (7 rows y) x (9 cols x) raster."""
    ys = np.round(rest[:, 1], 1)
    order = np.lexsort((rest[:, 0], ys))
    g = rest[order].reshape(GRID_R, GRID_C, 2)
    assert np.all(np.diff(g[..., 0], axis=1) > 5) and np.all(np.diff(g[..., 1], axis=0) > 5), 'marker grid not 7x9'
    return order


def affine_summary(d, valid, rest):
    """d (T, 63, 2), valid (T, 63) -> (T, 9): b_x, b_y, torsion, dilation, strain_1, strain_2, resid_rms, mean|d|,
    frac |d| > 1 px; least squares over valid markers only (d = A (r - rbar) + b)."""
    T = d.shape[0]
    out = np.zeros((T, 9), np.float32)
    rc = rest - rest.mean(0)
    X = np.concatenate([rc, np.ones((len(rc), 1))], 1)  # (63, 3)
    for t in range(T):
        v = valid[t]
        if v.sum() < 4:
            continue
        coef, *_ = np.linalg.lstsq(X[v], d[t, v], rcond=None)  # (3, 2): d_j = sum_k X_k coef[k, j]
        A = coef[:2].T  # (2, 2): d = A rc + b
        b = coef[2]
        res = d[t, v] - (rc[v] @ A.T + b)
        mag = np.linalg.norm(d[t, v], axis=-1)
        out[t] = [b[0], b[1], (A[1, 0] - A[0, 1]) / 2, (A[0, 0] + A[1, 1]) / 2, (A[0, 0] - A[1, 1]) / 2,
                  (A[0, 1] + A[1, 0]) / 2, np.sqrt((res ** 2).sum(-1).mean()), mag.mean(), (mag > 1.0).mean()]
    return out


def depth_summary(depth):
    """depth (T, 240, 320) canonical -> (T, 5): min depth, contact area, centroid x, centroid y, max press."""
    T = depth.shape[0]
    w = press_fullres(depth)
    dmin = depth.reshape(T, -1).min(1)
    area = (depth < CONTACT_THR).reshape(T, -1).mean(1)
    xs = (np.arange(W_IMG) + 0.5) / W_IMG * 2 - 1
    ys = (np.arange(H_IMG) + 0.5) / H_IMG * 2 - 1
    tot = w.reshape(T, -1).sum(1)
    cx = np.where(tot > 0, (w.sum(1) * xs[None]).sum(1) / np.maximum(tot, 1e-8), 0.0)
    cy = np.where(tot > 0, (w.sum(2) * ys[None]).sum(1) / np.maximum(tot, 1e-8), 0.0)
    return np.stack([dmin, area, cx, cy, w.reshape(T, -1).max(1)], 1).astype(np.float32)


def sanitize_m(m):
    """Clip the minimum depth (index 9) at 28.5 mm (zero indentation). Without contact the simulator reports the camera
    far plane (34.0), an arbitrary sentinel that would make m jump at every contact change."""
    m = m.copy() if isinstance(m, np.ndarray) else m.clone()
    m[..., 9] = np.minimum(m[..., 9], DEPTH_REF) if isinstance(m, np.ndarray) else m[..., 9].clamp(max=DEPTH_REF)
    return m


def normalize_m(m, m_valid, m_mean, m_std):
    """Standardised m: sentinel-sanitised, frames with < 50 % valid markers set to 0, clipped to +-M_CLIP.
    m (..., 14), m_valid (...)."""
    n = (sanitize_m(m) - m_mean) / m_std
    n = np.where(np.asarray(m_valid)[..., None], n, 0.0)
    return np.clip(n, -M_CLIP, M_CLIP).astype(np.float32)


def physics_summary(d, valid, rest, depth):
    """-> m (T, 14) in M_NAMES order, m_valid (T,), e_c (T,)."""
    aff = affine_summary(d, valid, rest)
    ds = depth_summary(depth)
    m = np.concatenate([aff, ds[:, :1], ds[:, 1:]], 1)  # 9 affine terms + min depth + 4 depth terms
    m_valid = valid.mean(1) >= 0.5
    e_c = ds[:, 0] < CONTACT_THR
    return m.astype(np.float32), m_valid, e_c


def process_gel(depth, marker, gel, rest=None):
    """Full per-gel processing of one episode (or one frame with T = 1).
    Returns dict(w96 (T,96,128) f32, mk (T,2,63,2) f32 canonical, rest (63,2), valid (T,63), d (T,63,2),
    m (T,14), m_valid (T,), e_c (T,), depth_min (T,))."""
    depth_c, mk = canonicalize(depth, marker, gel)
    if rest is None:
        rest = rest_grid(mk)
    valid, d = marker_validity(mk, rest)
    m, m_valid, e_c = physics_summary(d, valid, rest, depth_c)
    return dict(w96=press_field(depth_c), mk=mk, rest=rest.astype(np.float32), valid=valid, d=d, m=m,
                m_valid=m_valid, e_c=e_c, depth_min=m[:, 9].copy())


def deploy_frame(depth, marker, gel, stats):
    """One observation of one gel at deployment, processed like the training data.

    depth (240, 320); marker (2, 64, 2) [reference, current]; stats = the tactile statistics (``marker_order`` and
    ``rest_native`` per gel: the marker order is fixed by the sensor, the rest grid is the median over the training
    episodes). Returns w96 (96, 128) f32, d (7, 9, 2) displacement in raster order, valid (7, 9) bool,
    m (14,) with sanitize_m applied, m_valid, e_c."""
    order = np.asarray(stats['marker_order'][gel])
    rest_native = np.asarray(stats['rest_native'][gel], np.float32)
    rest_raster = np.asarray(stats['rest_grid'], np.float32)
    depth_c, mk = canonicalize(np.asarray(depth)[None], np.asarray(marker)[None], gel)
    valid, d = marker_validity(mk, rest_native)
    m, m_valid, e_c = physics_summary(d, valid, rest_native, depth_c)
    d_r = (mk[0, 1][order] - rest_raster).reshape(GRID_R, GRID_C, 2)
    return dict(w96=press_field(depth_c)[0], d=d_r, valid=valid[0][order].reshape(GRID_R, GRID_C),
                m=sanitize_m(m)[0], m_valid=bool(m_valid[0]), e_c=bool(e_c[0]))


# ----------------------------------------------------------------------------------------------------------------
# Torch field builder (training and deployment)
# ----------------------------------------------------------------------------------------------------------------
def grid_sampling_coords(rest_grid, device):
    """rest_grid (7, 9, 2) marker rest positions in pixels -> normalised sampling grid (1, 96, 128, 2) for
    grid_sample(align_corners=True) over the 7 x 9 marker grid."""
    xs = (torch.arange(OUT_W, device=device, dtype=torch.float32) + 0.5) * (W_IMG / OUT_W)
    ys = (torch.arange(OUT_H, device=device, dtype=torch.float32) + 0.5) * (H_IMG / OUT_H)
    x0, x1 = float(rest_grid[..., 0].min()), float(rest_grid[..., 0].max())
    y0, y1 = float(rest_grid[..., 1].min()), float(rest_grid[..., 1].max())
    gx = (xs - x0) / (x1 - x0) * 2 - 1
    gy = (ys - y0) / (y1 - y0) * 2 - 1
    g = torch.stack(torch.meshgrid(gy, gx, indexing='ij')[::-1], -1)  # (96, 128, 2) as (x, y)
    return g[None]


def inpaint_grid(d, m, iters=8):
    """d (N, 2, 7, 9) displacement, m (N, 1, 7, 9) {0, 1} validity -> d with invalid entries filled by iterated
    3x3 neighbour means of known entries (all-invalid samples stay 0)."""
    known = m.clone()
    d = d * m
    k = torch.ones(1, 1, 3, 3, device=d.device, dtype=d.dtype)
    for _ in range(iters):
        num = F.conv2d(d.reshape(-1, 1, GRID_R, GRID_C), k, padding=1).reshape(d.shape)
        den = F.conv2d(known, k, padding=1)
        fill = num / den.clamp(min=1e-6)
        newly = (den > 0) & (known == 0)
        d = torch.where(newly.expand_as(d), fill, d)
        known = torch.where(newly, torch.ones_like(known), known)
        if bool(known.min() > 0):
            break
    return d


def build_fields(w, d, valid, sigma_u, samp_grid, drop_frac: Optional[torch.Tensor] = None,
                 u_scale: Optional[torch.Tensor] = None):
    """Assemble the tactile observation X = (w, u_x / sigma_u, u_y / sigma_u).

    w (N, 96, 128) indentation field; d (N, 7, 9, 2) marker displacement in px (raster order); valid (N, 7, 9) bool;
    drop_frac (N,) optional fraction of markers dropped (augmentation); u_scale (N,) optional factor.
    Returns X (N, 3, 96, 128)."""
    N = w.shape[0]
    m = valid.float()
    if drop_frac is not None:
        keep = torch.rand(N, GRID_R, GRID_C, device=w.device) >= drop_frac[:, None, None]
        m = m * keep.float()
    dd = inpaint_grid(d.permute(0, 3, 1, 2).float(), m[:, None])  # (N, 2, 7, 9)
    u = F.grid_sample(dd, samp_grid.expand(N, -1, -1, -1), mode='bicubic', padding_mode='border', align_corners=True)
    u = u / sigma_u
    if u_scale is not None:
        u = u * u_scale[:, None, None, None]
    return torch.cat([w[:, None].float(), u], 1)
