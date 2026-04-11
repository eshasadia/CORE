"""
CORE – Ablation Study Script
============================
Systematically compares:

  A) CORE pipeline step ablation
     1. Baseline          – no registration (identity)
     2. Rigid-only        – Trimorph + XFeat rigid alignment
     3. Elastic-only      – deep multi-scale elastic registration (no rigid pre-step)
     4. CORE Coarse       – Rigid + Elastic (standard coarse pipeline)
     5. CORE Full         – Coarse + fine nuclei-level shape-aware registration
                           (requires --fixed-nuclei / --moving-nuclei CSVs)

  B) State-of-the-art comparison methods
     6.  Phase Correlation       – translation-only (scipy)
     7.  ITK Rigid (MI)          – SimpleITK Mattes Mutual Information rigid
     8.  ITK Demons              – SimpleITK Demons diffeomorphic non-rigid
     9.  ORB + RANSAC Affine     – OpenCV keypoint-based
     10. AKAZE + RANSAC Affine   – OpenCV keypoint-based (patent-free)

Metrics (computed whenever applicable)
  - TRE   – Target Registration Error (pixels, mean/median/std/max)
  - rTRE  – Relative TRE (mean/median)
  - NGF   – Normalised Gradient Field (higher = better edge alignment)
  - NCC   – Normalised Cross-Correlation (higher = better)
  - SSIM  – Structural Similarity Index (higher = better)
  - DICE  – Dice coefficient of tissue masks (higher = better)
  - Time  – per-method wall-clock time (seconds)

Usage
-----
  python ablation_study.py \\
      --source /path/to/moving.tiff \\
      --target /path/to/fixed.tiff \\
      [--fixed-landmarks  fixed_lm.csv] \\
      [--moving-landmarks moving_lm.csv] \\
      [--fixed-nuclei  fixed_nuclei.csv] \\
      [--moving-nuclei moving_nuclei.csv] \\
      [--output-dir    ./ablation_results] \\
      [--methods all] \\
      [--no-plots]

The landmark CSVs must have a header row; columns after the first are
treated as (x, y) coordinates (same format used elsewhere in the repo).
The nuclei CSVs must contain ``global_x`` and ``global_y`` columns.

Before running
--------------
  conda env create -f environment.yml
  conda activate core
  export VISION_AGENT_API_KEY="<your-key>"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import map_coordinates
from scipy.signal import correlate2d
from skimage import color
from skimage.metrics import structural_similarity as ssim

# ── Make sure the repo root is on sys.path when run directly ──────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.config import (
    PREPROCESSING_RESOLUTION,
    RegistrationParams,
)
from core.evaluation.evaluation import (
    apply_displacement_field_to_points,
    ngf_metric,
    rtre,
    tre,
)
from core.preprocessing.padding import pad_images
from core.preprocessing.preprocessing import extract_tissue_masks, load_wsi_images
from core.registration.nonrigid import elastic_image_registration
from core.registration.registration import (
    create_displacement_field,
    find_mutual_nearest_neighbors,
    perform_rigid_registration,
)
import core.utils.util as util

# ── Optional fine-registration imports ────────────────────────────────────────
try:
    from core.preprocessing.nuclei_analysis import load_nuclei_coordinates
    from core.preprocessing.padding import pad_landmarks
    from core.registration.nonrigid import compute_deformation_and_apply
    from core.registration.registration import perform_shape_aware_registration
    _FINE_REG_AVAILABLE = True
except ImportError:
    _FINE_REG_AVAILABLE = False

# ── Silence non-critical warnings ─────────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

ALL_METHODS = [
    "baseline",
    "rigid_only",
    "elastic_only",
    "core_coarse",
    "core_full",
    "phase_correlation",
    "itk_rigid_mi",
    "itk_demons",
    "orb_ransac",
    "akaze_ransac",
]

# Human-readable labels for tables / plots
METHOD_LABELS: Dict[str, str] = {
    "baseline":         "Baseline (none)",
    "rigid_only":       "CORE Rigid only",
    "elastic_only":     "CORE Elastic only",
    "core_coarse":      "CORE Coarse (R+E)",
    "core_full":        "CORE Full (R+E+Fine)",
    "phase_correlation": "Phase Correlation",
    "itk_rigid_mi":     "ITK Rigid (MI)",
    "itk_demons":       "ITK Demons",
    "orb_ransac":       "ORB + RANSAC",
    "akaze_ransac":     "AKAZE + RANSAC",
}


@dataclass
class MethodResult:
    """Holds all metric values for a single method."""

    method: str
    label: str
    time_total_s: float = float("nan")
    time_rigid_s: float = float("nan")
    time_elastic_s: float = float("nan")
    time_fine_s: float = float("nan")
    # Image-quality metrics
    ngf: float = float("nan")
    ncc: float = float("nan")
    ssim_score: float = float("nan")
    dice_mask: float = float("nan")
    # Landmark-based metrics (only when landmarks are provided)
    tre_mean: float = float("nan")
    tre_median: float = float("nan")
    tre_std: float = float("nan")
    tre_max: float = float("nan")
    rtre_mean: float = float("nan")
    rtre_median: float = float("nan")
    # Error / skip note
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "Method": self.label,
            "Time total (s)": _fmt(self.time_total_s),
            "Time rigid (s)": _fmt(self.time_rigid_s),
            "Time elastic (s)": _fmt(self.time_elastic_s),
            "Time fine (s)": _fmt(self.time_fine_s),
            "NGF ↑": _fmt(self.ngf, 4),
            "NCC ↑": _fmt(self.ncc, 4),
            "SSIM ↑": _fmt(self.ssim_score, 4),
            "DICE mask ↑": _fmt(self.dice_mask, 4),
            "TRE mean (px) ↓": _fmt(self.tre_mean, 2),
            "TRE median (px) ↓": _fmt(self.tre_median, 2),
            "TRE std (px)": _fmt(self.tre_std, 2),
            "TRE max (px) ↓": _fmt(self.tre_max, 2),
            "rTRE mean ↓": _fmt(self.rtre_mean, 5),
            "rTRE median ↓": _fmt(self.rtre_median, 5),
            "Note": self.note,
        }


def _fmt(val: float, decimals: int = 3) -> str:
    if np.isnan(val):
        return "N/A"
    return f"{val:.{decimals}f}"


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ncc(fixed: np.ndarray, moving: np.ndarray) -> float:
    """Normalised cross-correlation between two RGB images converted to grey."""
    fg = color.rgb2gray(fixed).astype(np.float64)
    mg = color.rgb2gray(moving).astype(np.float64)
    fg -= fg.mean(); fg_std = fg.std()
    mg -= mg.mean(); mg_std = mg.std()
    if fg_std < 1e-10 or mg_std < 1e-10:
        return 0.0
    return float(np.mean((fg / fg_std) * (mg / mg_std)))


def _ssim(fixed: np.ndarray, moving: np.ndarray) -> float:
    """SSIM on the grayscale versions (skimage)."""
    fg = color.rgb2gray(fixed)
    mg = color.rgb2gray(moving)
    data_range = max(fg.max(), mg.max()) - min(fg.min(), mg.min())
    if data_range < 1e-10:
        return 1.0
    return float(ssim(fg, mg, data_range=data_range))


def _dice_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Dice coefficient of two binary masks."""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    total = a.sum() + b.sum()
    if total == 0:
        return 0.0
    return float(2.0 * intersection / total)


def _compute_image_metrics(
    fixed: np.ndarray,
    warped: np.ndarray,
    fixed_mask: np.ndarray,
    warped_mask: np.ndarray,
    result: MethodResult,
) -> None:
    """Fill in NGF, NCC, SSIM, DICE on *result* in-place."""
    try:
        result.ngf = ngf_metric(fixed, warped)
    except Exception:
        pass
    try:
        result.ncc = _ncc(fixed, warped)
    except Exception:
        pass
    try:
        result.ssim_score = _ssim(fixed, warped)
    except Exception:
        pass
    try:
        result.dice_mask = _dice_masks(fixed_mask, warped_mask)
    except Exception:
        pass


def _compute_landmark_metrics(
    fixed_lm: np.ndarray,
    warped_lm: np.ndarray,
    image_shape: Tuple[int, int],
    result: MethodResult,
) -> None:
    """Fill in TRE / rTRE metrics on *result* in-place."""
    try:
        tre_vals = tre(fixed_lm, warped_lm)
        result.tre_mean   = float(np.mean(tre_vals))
        result.tre_median = float(np.median(tre_vals))
        result.tre_std    = float(np.std(tre_vals))
        result.tre_max    = float(np.max(tre_vals))
        rtre_vals = rtre(fixed_lm, warped_lm, image_shape[1], image_shape[0])
        result.rtre_mean   = float(np.mean(rtre_vals))
        result.rtre_median = float(np.median(rtre_vals))
    except Exception:
        pass


def _warp_landmarks_affine(landmarks: np.ndarray, matrix_3x3: np.ndarray) -> np.ndarray:
    """Apply a 3×3 affine matrix to (N,2) landmark array."""
    hom = np.hstack([landmarks, np.ones((len(landmarks), 1))])
    return (matrix_3x3 @ hom.T).T[:, :2]


def _warp_landmarks_displacement(
    landmarks: np.ndarray,
    disp_field_hw2: np.ndarray,
) -> np.ndarray:
    """
    Apply a (H, W, 2) displacement field to (N, 2) landmarks.
    Returns only the valid (in-bounds) subset; caller must handle size mismatch.
    """
    moved, valid = apply_displacement_field_to_points(landmarks, disp_field_hw2)
    # Re-insert invalid points with their original positions so shapes match
    out = landmarks.copy()
    out[valid] = moved
    return out


def _warp_mask(mask: np.ndarray, matrix_2x3: np.ndarray) -> np.ndarray:
    """Warp a binary mask with a 2×3 affine matrix."""
    h, w = mask.shape
    warped = cv2.warpAffine(
        mask.astype(np.uint8), matrix_2x3, (w, h), flags=cv2.INTER_NEAREST
    )
    return warped.astype(bool)


def _rgb_from_tensor(t) -> np.ndarray:
    """Convert PyTorch warped tensor (1,1,H,W) or (1,C,H,W) to uint8 RGB ndarray."""
    arr = t.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[0] == 1:          # grayscale → RGB
        arr = np.concatenate([arr, arr, arr], axis=0)
    arr = np.transpose(arr, (1, 2, 0))   # CHW → HWC
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Registration method implementations
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Baseline ───────────────────────────────────────────────────────────────

def run_baseline(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    fixed_lm: Optional[np.ndarray],
    moving_lm: Optional[np.ndarray],
) -> MethodResult:
    result = MethodResult(method="baseline", label=METHOD_LABELS["baseline"])
    t0 = time.perf_counter()
    warped = source.copy()
    result.time_total_s = time.perf_counter() - t0
    _compute_image_metrics(target, warped, target_mask, source_mask, result)
    if fixed_lm is not None and moving_lm is not None:
        _compute_landmark_metrics(fixed_lm, moving_lm, target.shape, result)
    return result


# ── 2. CORE Rigid-only ────────────────────────────────────────────────────────

def run_rigid_only(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    fixed_lm: Optional[np.ndarray],
    moving_lm: Optional[np.ndarray],
) -> MethodResult:
    result = MethodResult(method="rigid_only", label=METHOD_LABELS["rigid_only"])
    t0 = time.perf_counter()
    try:
        warped, transform = perform_rigid_registration(
            source, target, source_mask, target_mask
        )
        result.time_rigid_s = time.perf_counter() - t0
        result.time_total_s = result.time_rigid_s
        warped_mask = _warp_mask(source_mask, transform[:2])
        _compute_image_metrics(target, warped, target_mask, warped_mask, result)
        if fixed_lm is not None and moving_lm is not None:
            warped_lm = _warp_landmarks_affine(moving_lm, np.linalg.inv(transform))
            _compute_landmark_metrics(fixed_lm, warped_lm, target.shape, result)
    except Exception as exc:
        result.note = str(exc)
    return result


# ── 3. CORE Elastic-only ──────────────────────────────────────────────────────

def run_elastic_only(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    fixed_lm: Optional[np.ndarray],
    moving_lm: Optional[np.ndarray],
) -> MethodResult:
    result = MethodResult(method="elastic_only", label=METHOD_LABELS["elastic_only"])
    t0 = time.perf_counter()
    try:
        def_field, warped_t = elastic_image_registration(source, target)
        result.time_elastic_s = time.perf_counter() - t0
        result.time_total_s = result.time_elastic_s
        warped = _rgb_from_tensor(warped_t)
        # Warp mask using the displacement field
        disp_np = def_field.detach().cpu().numpy()[0]  # (H, W, 2) in normalised coords
        disp_hw2 = _denorm_displacement(disp_np, source.shape[:2])
        warped_mask = _warp_mask_displacement(source_mask, disp_hw2)
        _compute_image_metrics(target, warped, target_mask, warped_mask, result)
        if fixed_lm is not None and moving_lm is not None:
            warped_lm = _warp_landmarks_displacement(moving_lm, disp_hw2)
            _compute_landmark_metrics(fixed_lm, warped_lm, target.shape, result)
    except Exception as exc:
        result.note = str(exc)
    return result


# ── 4. CORE Coarse (Rigid + Elastic) ─────────────────────────────────────────

def run_core_coarse(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    fixed_lm: Optional[np.ndarray],
    moving_lm: Optional[np.ndarray],
) -> MethodResult:
    result = MethodResult(method="core_coarse", label=METHOD_LABELS["core_coarse"])
    t0 = time.perf_counter()
    try:
        # Rigid step
        t_rigid0 = time.perf_counter()
        rigid_warped, transform = perform_rigid_registration(
            source, target, source_mask, target_mask
        )
        result.time_rigid_s = time.perf_counter() - t_rigid0

        # Elastic step
        t_elast0 = time.perf_counter()
        def_field, warped_t = elastic_image_registration(rigid_warped, target)
        result.time_elastic_s = time.perf_counter() - t_elast0
        result.time_total_s = time.perf_counter() - t0

        warped = _rgb_from_tensor(warped_t)
        disp_np = def_field.detach().cpu().numpy()[0]
        disp_hw2 = _denorm_displacement(disp_np, rigid_warped.shape[:2])

        # Compose: first rigid, then elastic
        rigid_warped_mask = _warp_mask(source_mask, transform[:2])
        warped_mask = _warp_mask_displacement(rigid_warped_mask, disp_hw2)
        _compute_image_metrics(target, warped, target_mask, warped_mask, result)

        if fixed_lm is not None and moving_lm is not None:
            rigid_lm = _warp_landmarks_affine(moving_lm, np.linalg.inv(transform))
            final_lm = _warp_landmarks_displacement(rigid_lm, disp_hw2)
            _compute_landmark_metrics(fixed_lm, final_lm, target.shape, result)
    except Exception as exc:
        result.note = str(exc)
    return result


# ── 5. CORE Full (Coarse + Fine nuclei) ──────────────────────────────────────

def run_core_full(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    fixed_lm: Optional[np.ndarray],
    moving_lm: Optional[np.ndarray],
    fixed_nuclei_csv: Optional[str],
    moving_nuclei_csv: Optional[str],
    padding_params: dict,
) -> MethodResult:
    result = MethodResult(method="core_full", label=METHOD_LABELS["core_full"])

    if not _FINE_REG_AVAILABLE:
        result.note = "Fine registration modules not importable; skipped."
        return result

    if fixed_nuclei_csv is None or moving_nuclei_csv is None:
        result.note = "Nuclei CSVs not provided; skipped."
        return result

    t0 = time.perf_counter()
    try:
        # Coarse rigid
        t_r0 = time.perf_counter()
        rigid_warped, transform = perform_rigid_registration(
            source, target, source_mask, target_mask
        )
        result.time_rigid_s = time.perf_counter() - t_r0

        # Coarse elastic
        t_e0 = time.perf_counter()
        def_field, warped_t = elastic_image_registration(rigid_warped, target)
        result.time_elastic_s = time.perf_counter() - t_e0

        # Convert displacement field for downstream use
        disp_field_np = util.tc_df_to_np_df(def_field)

        # Fine nuclei registration
        t_f0 = time.perf_counter()
        moving_df = load_nuclei_coordinates(moving_nuclei_csv)
        fixed_df  = load_nuclei_coordinates(fixed_nuclei_csv)

        deform_field, moving_updated, fixed_pts, _ = compute_deformation_and_apply(
            source,
            transform,
            def_field,
            moving_df,
            fixed_df,
            padding_params,
            util,
            pad_landmarks,
        )

        _, _, shape_transformed = perform_shape_aware_registration(
            fixed_pts,
            moving_updated,
            shape_weight=0.3,
            max_iterations=100,
            tolerance=1e-11,
        )
        result.time_fine_s = time.perf_counter() - t_f0
        result.time_total_s = time.perf_counter() - t0

        # Image quality from elastic warped result
        warped = _rgb_from_tensor(warped_t)
        disp_np = def_field.detach().cpu().numpy()[0]
        disp_hw2 = _denorm_displacement(disp_np, rigid_warped.shape[:2])
        rigid_warped_mask = _warp_mask(source_mask, transform[:2])
        warped_mask = _warp_mask_displacement(rigid_warped_mask, disp_hw2)
        _compute_image_metrics(target, warped, target_mask, warped_mask, result)

        # Landmark metrics using shape-aware output
        if fixed_lm is not None:
            _compute_landmark_metrics(fixed_pts, shape_transformed, target.shape, result)
    except Exception as exc:
        result.note = str(exc)
    return result


# ── 6. Phase Correlation ──────────────────────────────────────────────────────

def run_phase_correlation(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    fixed_lm: Optional[np.ndarray],
    moving_lm: Optional[np.ndarray],
) -> MethodResult:
    from skimage.registration import phase_cross_correlation

    result = MethodResult(
        method="phase_correlation", label=METHOD_LABELS["phase_correlation"]
    )
    t0 = time.perf_counter()
    try:
        src_g = color.rgb2gray(source)
        tgt_g = color.rgb2gray(target)
        shift, _, _ = phase_cross_correlation(tgt_g, src_g, upsample_factor=10)
        # Build a translation-only affine matrix
        M = np.array([[1, 0, shift[1]], [0, 1, shift[0]]], dtype=np.float64)
        warped = cv2.warpAffine(source, M, (target.shape[1], target.shape[0]))
        result.time_total_s = time.perf_counter() - t0
        warped_mask = _warp_mask(source_mask, M)
        _compute_image_metrics(target, warped, target_mask, warped_mask, result)
        if fixed_lm is not None and moving_lm is not None:
            M3 = np.vstack([M, [0, 0, 1]])
            warped_lm = _warp_landmarks_affine(moving_lm, np.linalg.inv(M3))
            _compute_landmark_metrics(fixed_lm, warped_lm, target.shape, result)
    except Exception as exc:
        result.note = str(exc)
    return result


# ── 7. ITK Rigid (Mutual Information) ────────────────────────────────────────

def run_itk_rigid_mi(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    fixed_lm: Optional[np.ndarray],
    moving_lm: Optional[np.ndarray],
) -> MethodResult:
    result = MethodResult(method="itk_rigid_mi", label=METHOD_LABELS["itk_rigid_mi"])
    t0 = time.perf_counter()
    try:
        fixed_sitk  = _rgb_to_sitk(target)
        moving_sitk = _rgb_to_sitk(source)

        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.1)
        reg.SetInterpolator(sitk.sitkLinear)

        # Initialise with moments
        init = sitk.CenteredTransformInitializer(
            fixed_sitk, moving_sitk,
            sitk.Euler2DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS,
        )
        reg.SetInitialTransform(init)
        reg.SetOptimizerAsGradientDescent(
            learningRate=1.0, numberOfIterations=200,
            convergenceMinimumValue=1e-6, convergenceWindowSize=10,
        )
        reg.SetOptimizerScalesFromPhysicalShift()

        final_transform = reg.Execute(
            sitk.Cast(fixed_sitk, sitk.sitkFloat32),
            sitk.Cast(moving_sitk, sitk.sitkFloat32),
        )

        warped_sitk = sitk.Resample(
            moving_sitk, fixed_sitk, final_transform,
            sitk.sitkLinear, 0.0, moving_sitk.GetPixelID(),
        )
        warped = sitk.GetArrayFromImage(warped_sitk)
        if warped.ndim == 2:
            warped = np.stack([warped] * 3, axis=-1)
        warped = warped.astype(np.uint8)

        result.time_total_s = time.perf_counter() - t0

        # Approximate mask warp via ITK transform
        warped_mask = _warp_mask_sitk(source_mask, fixed_sitk, final_transform)
        _compute_image_metrics(target, warped, target_mask, warped_mask, result)

        if fixed_lm is not None and moving_lm is not None:
            warped_lm = _transform_landmarks_sitk(moving_lm, final_transform, source.shape)
            _compute_landmark_metrics(fixed_lm, warped_lm, target.shape, result)
    except Exception as exc:
        result.note = str(exc)
    return result


# ── 8. ITK Demons ─────────────────────────────────────────────────────────────

def run_itk_demons(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    fixed_lm: Optional[np.ndarray],
    moving_lm: Optional[np.ndarray],
) -> MethodResult:
    result = MethodResult(method="itk_demons", label=METHOD_LABELS["itk_demons"])
    t0 = time.perf_counter()
    try:
        fixed_sitk  = sitk.Cast(_rgb_to_sitk(target),  sitk.sitkFloat32)
        moving_sitk = sitk.Cast(_rgb_to_sitk(source), sitk.sitkFloat32)

        demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
        demons.SetNumberOfIterations(200)
        demons.SetStandardDeviations(1.5)
        disp_field = demons.Execute(fixed_sitk, moving_sitk)

        disp_tx = sitk.DisplacementFieldTransform(disp_field)
        warped_sitk = sitk.Resample(
            sitk.Cast(_rgb_to_sitk(source), sitk.sitkFloat32),
            fixed_sitk, disp_tx,
            sitk.sitkLinear, 0.0, sitk.sitkFloat32,
        )
        warped = sitk.GetArrayFromImage(sitk.Cast(warped_sitk, sitk.sitkUInt8))
        if warped.ndim == 2:
            warped = np.stack([warped] * 3, axis=-1)

        result.time_total_s = time.perf_counter() - t0

        disp_np = sitk.GetArrayFromImage(disp_field)  # (H, W, 2)
        warped_mask = _warp_mask_displacement(source_mask, disp_np)
        _compute_image_metrics(target, warped, target_mask, warped_mask, result)

        if fixed_lm is not None and moving_lm is not None:
            warped_lm = _warp_landmarks_displacement(moving_lm, disp_np)
            _compute_landmark_metrics(fixed_lm, warped_lm, target.shape, result)
    except Exception as exc:
        result.note = str(exc)
    return result


# ── 9. ORB + RANSAC ───────────────────────────────────────────────────────────

def run_orb_ransac(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    fixed_lm: Optional[np.ndarray],
    moving_lm: Optional[np.ndarray],
) -> MethodResult:
    result = MethodResult(method="orb_ransac", label=METHOD_LABELS["orb_ransac"])
    t0 = time.perf_counter()
    try:
        M = _keypoint_affine(source, target, descriptor="ORB")
        warped = cv2.warpAffine(source, M, (target.shape[1], target.shape[0]))
        result.time_total_s = time.perf_counter() - t0
        warped_mask = _warp_mask(source_mask, M)
        _compute_image_metrics(target, warped, target_mask, warped_mask, result)
        if fixed_lm is not None and moving_lm is not None:
            M3 = np.vstack([M, [0, 0, 1]])
            warped_lm = _warp_landmarks_affine(moving_lm, np.linalg.inv(M3))
            _compute_landmark_metrics(fixed_lm, warped_lm, target.shape, result)
    except Exception as exc:
        result.note = str(exc)
    return result


# ── 10. AKAZE + RANSAC ────────────────────────────────────────────────────────

def run_akaze_ransac(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    fixed_lm: Optional[np.ndarray],
    moving_lm: Optional[np.ndarray],
) -> MethodResult:
    result = MethodResult(method="akaze_ransac", label=METHOD_LABELS["akaze_ransac"])
    t0 = time.perf_counter()
    try:
        M = _keypoint_affine(source, target, descriptor="AKAZE")
        warped = cv2.warpAffine(source, M, (target.shape[1], target.shape[0]))
        result.time_total_s = time.perf_counter() - t0
        warped_mask = _warp_mask(source_mask, M)
        _compute_image_metrics(target, warped, target_mask, warped_mask, result)
        if fixed_lm is not None and moving_lm is not None:
            M3 = np.vstack([M, [0, 0, 1]])
            warped_lm = _warp_landmarks_affine(moving_lm, np.linalg.inv(M3))
            _compute_landmark_metrics(fixed_lm, warped_lm, target.shape, result)
    except Exception as exc:
        result.note = str(exc)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Shared low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _keypoint_affine(
    source: np.ndarray,
    target: np.ndarray,
    descriptor: str = "ORB",
    min_matches: int = 10,
) -> np.ndarray:
    """
    Estimate a 2×3 affine matrix via keypoint matching + RANSAC.

    Parameters
    ----------
    source, target : RGB uint8 arrays
    descriptor     : 'ORB' or 'AKAZE'
    min_matches    : minimum number of inlier matches required

    Returns
    -------
    M : (2, 3) float64 affine matrix
    """
    src_g = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    tgt_g = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)

    if descriptor == "ORB":
        det = cv2.ORB_create(nfeatures=10_000)
        norm = cv2.NORM_HAMMING
    else:  # AKAZE
        det = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING

    kp1, des1 = det.detectAndCompute(src_g, None)
    kp2, des2 = det.detectAndCompute(tgt_g, None)

    if des1 is None or des2 is None or len(des1) < min_matches or len(des2) < min_matches:
        raise RuntimeError(
            f"{descriptor}: insufficient keypoints "
            f"(src={len(kp1) if kp1 else 0}, tgt={len(kp2) if kp2 else 0})"
        )

    bf = cv2.BFMatcher(norm, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    if len(matches) < min_matches:
        raise RuntimeError(
            f"{descriptor}: too few matches ({len(matches)} < {min_matches})"
        )

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        raise RuntimeError(f"{descriptor}: RANSAC affine estimation failed.")
    return M.astype(np.float64)


def _rgb_to_sitk(img: np.ndarray) -> sitk.Image:
    """Convert RGB uint8 array to grayscale SimpleITK image."""
    gray = (color.rgb2gray(img) * 255).astype(np.uint8)
    return sitk.GetImageFromArray(gray)


def _warp_mask_sitk(
    mask: np.ndarray,
    reference_sitk: sitk.Image,
    transform: sitk.Transform,
) -> np.ndarray:
    """Warp a binary mask using a SimpleITK transform."""
    mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
    mask_sitk.CopyInformation(reference_sitk)
    warped = sitk.Resample(
        mask_sitk, reference_sitk, transform,
        sitk.sitkNearestNeighbor, 0, mask_sitk.GetPixelID(),
    )
    return sitk.GetArrayFromImage(warped).astype(bool)


def _transform_landmarks_sitk(
    landmarks: np.ndarray,
    transform: sitk.Transform,
    image_shape: Tuple[int, ...],
) -> np.ndarray:
    """
    Apply a SimpleITK transform to (N, 2) landmark coordinates.
    SimpleITK transforms operate in physical space; we assume unit spacing.
    The transform maps fixed → moving (inverse direction), so we use the
    inverse to map moving → fixed.
    """
    inv_tx = transform.GetInverse()
    out = np.zeros_like(landmarks)
    for i, (x, y) in enumerate(landmarks):
        tx, ty = inv_tx.TransformPoint((float(x), float(y)))
        out[i] = [tx, ty]
    return out


def _denorm_displacement(
    disp_norm: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Convert a PyTorch grid-sample normalised displacement field (H, W, 2) in
    [-1, 1] to pixel-space displacement (H, W, 2).
    """
    H, W = image_shape
    out = disp_norm.copy()
    out[..., 0] = disp_norm[..., 0] * (W / 2.0)   # x dimension
    out[..., 1] = disp_norm[..., 1] * (H / 2.0)   # y dimension
    return out


def _warp_mask_displacement(
    mask: np.ndarray,
    disp_hw2: np.ndarray,
) -> np.ndarray:
    """Apply a pixel-space (H, W, 2) displacement field to warp a binary mask."""
    h, w = mask.shape
    map_x = (np.arange(w, dtype=np.float32)[None, :] + disp_hw2[..., 0]).astype(np.float32)
    map_y = (np.arange(h, dtype=np.float32)[:, None] + disp_hw2[..., 1]).astype(np.float32)
    warped = cv2.remap(
        mask.astype(np.uint8), map_x, map_y,
        cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return warped.astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# Landmark loading
# ─────────────────────────────────────────────────────────────────────────────

def load_landmarks(path: str) -> np.ndarray:
    """
    Load landmarks from a CSV file.
    Expects a header row; coordinates are taken from the first two numeric
    columns after any index column.

    Returns
    -------
    np.ndarray of shape (N, 2)
    """
    df = pd.read_csv(path)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        raise ValueError(
            f"Landmark file {path!r} must contain at least two numeric columns."
        )
    return df[num_cols[:2]].values.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _results_to_dataframe(results: List[MethodResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in results])


def print_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 120)
    print("ABLATION STUDY RESULTS")
    print("=" * 120)
    print(df.to_string(index=False))
    print("=" * 120 + "\n")


def save_csv(df: pd.DataFrame, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "ablation_results.csv")
    df.to_csv(path, index=False)
    print(f"Results saved to: {path}")
    return path


def plot_results(
    results: List[MethodResult],
    output_dir: str,
    has_landmarks: bool,
) -> None:
    """Create and save bar-chart summaries for key metrics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available; skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    labels = [r.label for r in results]

    # ── Image-quality metrics ──────────────────────────────────────────────
    img_metrics = {
        "NGF ↑": [r.ngf for r in results],
        "NCC ↑": [r.ncc for r in results],
        "SSIM ↑": [r.ssim_score for r in results],
        "DICE mask ↑": [r.dice_mask for r in results],
    }
    _bar_chart(
        labels, img_metrics,
        title="Image-Quality Metrics",
        ylabel="Score (higher is better)",
        path=os.path.join(output_dir, "ablation_image_quality.png"),
    )

    # ── Timing ────────────────────────────────────────────────────────────
    time_data = {
        "Total": [r.time_total_s for r in results],
        "Rigid": [r.time_rigid_s for r in results],
        "Elastic": [r.time_elastic_s for r in results],
        "Fine": [r.time_fine_s for r in results],
    }
    _bar_chart(
        labels, time_data,
        title="Runtime per Stage (seconds)",
        ylabel="Time (s)",
        path=os.path.join(output_dir, "ablation_runtime.png"),
        fill_nan=0.0,
    )

    # ── Landmark metrics (optional) ───────────────────────────────────────
    if has_landmarks:
        lm_metrics = {
            "TRE mean (px) ↓": [r.tre_mean for r in results],
            "TRE median (px) ↓": [r.tre_median for r in results],
            "rTRE mean ↓": [r.rtre_mean for r in results],
        }
        _bar_chart(
            labels, lm_metrics,
            title="Landmark Registration Error",
            ylabel="Error (lower is better)",
            path=os.path.join(output_dir, "ablation_landmark_error.png"),
        )

    print(f"Plots saved to: {output_dir}")


def _bar_chart(
    labels: List[str],
    series: Dict[str, List[float]],
    title: str,
    ylabel: str,
    path: str,
    fill_nan: float = float("nan"),
) -> None:
    import matplotlib.pyplot as plt

    n_groups = len(labels)
    n_series = len(series)
    x = np.arange(n_groups)
    width = 0.8 / max(n_series, 1)

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 1.5), 5))
    for i, (name, vals) in enumerate(series.items()):
        vals_clean = [
            fill_nan if np.isnan(v) else v
            for v in vals
        ]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals_clean, width, label=name)

    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(args: argparse.Namespace) -> pd.DataFrame:
    # ── Load and preprocess images ────────────────────────────────────────
    print("── Loading WSI images ───────────────────────────────────────")
    source_wsi, target_wsi, source, target = load_wsi_images(
        args.source, args.target, PREPROCESSING_RESOLUTION
    )
    print(f"  Source shape : {source.shape}")
    print(f"  Target shape : {target.shape}")

    print("\n── Padding images ───────────────────────────────────────────")
    source_prep, target_prep, padding_params = pad_images(source, target)
    print(f"  Padded shape : {source_prep.shape}")

    print("\n── Extracting tissue masks ──────────────────────────────────")
    source_mask, target_mask = extract_tissue_masks(
        source_prep, target_prep, artefacts=False
    )
    print("  Tissue masks extracted ✓")

    # ── Load landmarks if provided ────────────────────────────────────────
    fixed_lm = moving_lm = None
    if args.fixed_landmarks and args.moving_landmarks:
        print("\n── Loading landmarks ────────────────────────────────────────")
        fixed_lm  = load_landmarks(args.fixed_landmarks)
        moving_lm = load_landmarks(args.moving_landmarks)
        print(f"  Fixed : {len(fixed_lm)} pts   Moving : {len(moving_lm)} pts")
    else:
        print("\n[INFO] No landmark files provided; TRE/rTRE will be skipped.")

    # ── Determine which methods to run ────────────────────────────────────
    methods_to_run: List[str]
    if args.methods == ["all"]:
        methods_to_run = ALL_METHODS
    else:
        methods_to_run = [m for m in args.methods if m in ALL_METHODS]
        unknown = set(args.methods) - set(ALL_METHODS)
        if unknown:
            print(f"[WARN] Unknown methods ignored: {unknown}")

    # ── Dispatch ──────────────────────────────────────────────────────────
    results: List[MethodResult] = []
    shared_kwargs = dict(
        source=source_prep,
        target=target_prep,
        source_mask=source_mask,
        target_mask=target_mask,
        fixed_lm=fixed_lm,
        moving_lm=moving_lm,
    )

    runners = {
        "baseline":          lambda: run_baseline(**shared_kwargs),
        "rigid_only":        lambda: run_rigid_only(**shared_kwargs),
        "elastic_only":      lambda: run_elastic_only(**shared_kwargs),
        "core_coarse":       lambda: run_core_coarse(**shared_kwargs),
        "core_full":         lambda: run_core_full(
            **shared_kwargs,
            fixed_nuclei_csv=args.fixed_nuclei,
            moving_nuclei_csv=args.moving_nuclei,
            padding_params=padding_params,
        ),
        "phase_correlation": lambda: run_phase_correlation(**shared_kwargs),
        "itk_rigid_mi":      lambda: run_itk_rigid_mi(**shared_kwargs),
        "itk_demons":        lambda: run_itk_demons(**shared_kwargs),
        "orb_ransac":        lambda: run_orb_ransac(**shared_kwargs),
        "akaze_ransac":      lambda: run_akaze_ransac(**shared_kwargs),
    }

    for method_key in methods_to_run:
        label = METHOD_LABELS.get(method_key, method_key)
        print(f"\n── Running : {label} {'─' * max(1, 60 - len(label))}")
        try:
            result = runners[method_key]()
        except Exception as exc:
            result = MethodResult(
                method=method_key,
                label=label,
                note=f"UNEXPECTED ERROR: {exc}",
            )
        results.append(result)
        note = f"  [{result.note}]" if result.note else ""
        print(
            f"  Done in {result.time_total_s:.1f}s  |  "
            f"NGF={_fmt(result.ngf, 4)}  NCC={_fmt(result.ncc, 4)}  "
            f"SSIM={_fmt(result.ssim_score, 4)}  DICE={_fmt(result.dice_mask, 4)}"
            + (f"  TRE={_fmt(result.tre_mean, 2)}px" if fixed_lm is not None else "")
            + note
        )

    # ── Output ────────────────────────────────────────────────────────────
    df = _results_to_dataframe(results)
    print_table(df)
    save_csv(df, args.output_dir)

    if not args.no_plots:
        plot_results(results, args.output_dir, has_landmarks=(fixed_lm is not None))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ablation_study.py",
        description="CORE registration ablation study vs. SOTA methods.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--source", required=True,
        help="Path to the source (moving) WSI file.",
    )
    p.add_argument(
        "--target", required=True,
        help="Path to the target (fixed) WSI file.",
    )
    p.add_argument(
        "--fixed-landmarks", default=None,
        help=(
            "CSV of fixed (target) landmark coordinates.\n"
            "Header row required; first two numeric columns used as (x, y)."
        ),
    )
    p.add_argument(
        "--moving-landmarks", default=None,
        help="CSV of moving (source) landmark coordinates (same format).",
    )
    p.add_argument(
        "--fixed-nuclei", default=None,
        help=(
            "CSV of fixed nuclei (required for 'core_full').\n"
            "Must contain 'global_x' and 'global_y' columns."
        ),
    )
    p.add_argument(
        "--moving-nuclei", default=None,
        help="CSV of moving nuclei (required for 'core_full').",
    )
    p.add_argument(
        "--output-dir", default="./ablation_results",
        help="Directory where CSV and plots are saved (default: ./ablation_results).",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        choices=ALL_METHODS + ["all"],
        metavar="METHOD",
        help=(
            "Which methods to run. Pass 'all' (default) or a subset:\n"
            "  " + "  ".join(ALL_METHODS)
        ),
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Suppress matplotlib output.",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
