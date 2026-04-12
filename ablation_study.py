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

  C) Point-based methods
     11. SIFT + RANSAC Affine    – OpenCV SIFT keypoint-based
     12. ICP (Open3D)            – Iterative Closest Point on tissue masks
     13. CPD Rigid               – Coherent Point Drift (rigid)
     14. CPD Affine              – Coherent Point Drift (affine)
     15. CPD Non-rigid           – Coherent Point Drift (deformable)

  D) External registration toolboxes (require binaries on PATH)
     16. Elastix Affine          – elastix + AdvancedMattesMutualInformation affine
     17. Elastix B-Spline        – elastix + B-spline non-rigid (FinalGridSpacing=16 px)
     18. NiftyReg Affine         – reg_aladin rigid/affine
     19. NiftyReg B-Spline       – reg_aladin (init) → reg_f3d free-form deformation
     20. DROP Deformable         – DROP2/DROP discrete-optimisation deformable

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

  # Optional: install external toolboxes for methods 16–20
  #   Elastix : https://elastix.lumc.nl/  (elastix + transformix on PATH)
  #   NiftyReg: https://github.com/KCL-BMEIS/niftyreg  (reg_aladin + reg_f3d on PATH)
  #   DROP    : https://www.mrf-registration.net/deformable/  (DROP2 or DROP on PATH)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
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
    convert_4x4_to_3x3_transform,
    create_displacement_field,
    find_mutual_nearest_neighbors,
    perform_cpd_registration,
    perform_icp_registration,
    perform_rigid_registration,
)
from core.registration.cpd import CPD as _CPD
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
    # ── Point-based methods ──────────────────────────────────────────────
    "sift_ransac",
    "icp",
    "cpd_rigid",
    "cpd_affine",
    "cpd_nonrigid",
    # ── External registration toolboxes ─────────────────────────────────
    "elastix_affine",
    "elastix_bspline",
    "niftyreg_affine",
    "niftyreg_f3d",
    "drop",
]

# Human-readable labels for tables / plots
METHOD_LABELS: Dict[str, str] = {
    "baseline":          "Baseline (none)",
    "rigid_only":        "CORE Rigid only",
    "elastic_only":      "CORE Elastic only",
    "core_coarse":       "CORE Coarse (R+E)",
    "core_full":         "CORE Full (R+E+Fine)",
    "phase_correlation": "Phase Correlation",
    "itk_rigid_mi":      "ITK Rigid (MI)",
    "itk_demons":        "ITK Demons",
    "orb_ransac":        "ORB + RANSAC",
    "akaze_ransac":      "AKAZE + RANSAC",
    # ── Point-based methods ──────────────────────────────────────────────
    "sift_ransac":       "SIFT + RANSAC",
    "icp":               "ICP (Open3D)",
    "cpd_rigid":         "CPD Rigid",
    "cpd_affine":        "CPD Affine",
    "cpd_nonrigid":      "CPD Non-rigid",
    # ── External registration toolboxes ─────────────────────────────────
    "elastix_affine":    "Elastix Affine",
    "elastix_bspline":   "Elastix B-Spline",
    "niftyreg_affine":   "NiftyReg Affine (reg_aladin)",
    "niftyreg_f3d":      "NiftyReg B-Spline (reg_f3d)",
    "drop":              "DROP Deformable",
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
    fg -= fg.mean()
    fg_std = fg.std()
    mg -= mg.mean()
    mg_std = mg.std()
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
    descriptor     : 'ORB', 'AKAZE', or 'SIFT'
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
    elif descriptor == "SIFT":
        det = cv2.SIFT_create()
        norm = cv2.NORM_L2
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
# Helpers shared by external-tool runners (Elastix, NiftyReg, DROP)
# ─────────────────────────────────────────────────────────────────────────────

def _save_mhd(img_rgb: np.ndarray, path: str) -> None:
    """Write an RGB uint8 array to disk as a grayscale float32 MHD/MHA image."""
    gray = (color.rgb2gray(img_rgb) * 255.0).astype(np.float32)
    sitk.WriteImage(sitk.GetImageFromArray(gray), path)


def _load_warped_mhd(path: str) -> np.ndarray:
    """
    Load a registered grayscale image (any SimpleITK-readable format) and
    return it as a uint8 RGB ndarray suitable for metric computation.
    """
    sitk_img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    if arr.ndim == 3:           # strip leading singleton axis if present
        arr = arr[0]
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    arr = arr.clip(0, 255).astype(np.uint8)
    return np.stack([arr, arr, arr], axis=-1)


def _run_cmd(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """
    Run *cmd* as a subprocess and return ``(returncode, stdout, stderr)``.
    Raises ``RuntimeError`` if the executable is not found on PATH.
    """
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=cwd,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        raise RuntimeError(
            f"Executable not found: {cmd[0]!r}. "
            "Please install the tool and ensure it is on your PATH."
        )


def _load_mhd_displacement(path: str) -> np.ndarray:
    """
    Load a vector-field image (H, W, 2) from *path* via SimpleITK.
    Returns a float32 ndarray in pixel-space (x, y) order.
    """
    sitk_field = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(sitk_field).astype(np.float32)
    # SimpleITK stores vector images as (H, W, nComponents) after GetArrayFromImage
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        return arr[..., :2]
    raise RuntimeError(
        f"Unexpected displacement field shape {arr.shape} in {path!r}."
    )


def _read_niftyreg_affine_matrix(path: str) -> np.ndarray:
    """
    Read a NiftyReg affine text file (4×4 matrix) and return the upper-left
    3×3 sub-matrix suitable for ``_warp_landmarks_affine``.
    """
    mat = np.loadtxt(path)   # (4, 4)
    return mat[:3, :3].copy()


def _niftyreg_affine_to_2x3(path: str) -> np.ndarray:
    """
    Read a NiftyReg affine text file and return the 2×3 OpenCV warpAffine
    matrix that maps source → target coordinates.
    """
    mat4x4 = np.loadtxt(path)  # (4, 4)
    # NiftyReg stores world-space affine; assume unit voxel spacing (pixel = voxel)
    return mat4x4[:2, [0, 1, 3]].astype(np.float64)   # rows 0-1, cols x/y/tx


# ─────────────────────────────────────────────────────────────────────────────
# Point-based method helpers
# ─────────────────────────────────────────────────────────────────────────────

_PT_MAX_POINTS = 3000   # max tissue points used by ICP / CPD


def _extract_tissue_points(mask: np.ndarray, max_points: int = _PT_MAX_POINTS) -> np.ndarray:
    """
    Subsample foreground pixel coordinates from a binary tissue mask.

    Returns
    -------
    np.ndarray of shape (N, 2) with (x, y) float32 coordinates.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("Tissue mask is empty — cannot extract points.")
    points = np.column_stack([xs, ys]).astype(np.float32)
    if len(points) > max_points:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
    return points


def _cpd_rigid_to_2x3(R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    """
    Convert CPD rigid parameters (R, t, s) to a 2×3 OpenCV affine matrix.

    CPD rigid transform:  T(y) = s * y @ R.T + t
    Expanded for a row vector [x, y]:
        x' = s*(R[0,0]*x + R[1,0]*y) + t[0]
        y' = s*(R[0,1]*x + R[1,1]*y) + t[1]
    """
    A = s * R          # 2×2
    return np.array([
        [A[0, 0], A[1, 0], t[0]],
        [A[0, 1], A[1, 1], t[1]],
    ], dtype=np.float64)


def _cpd_affine_to_2x3(B: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Convert CPD affine parameters (B, t) to a 2×3 OpenCV affine matrix.

    CPD affine transform:  T(y) = y @ B.T + t
    Expanded:
        x' = B[0,0]*x + B[1,0]*y + t[0]
        y' = B[0,1]*x + B[1,1]*y + t[1]
    """
    return np.array([
        [B[0, 0], B[1, 0], t[0]],
        [B[0, 1], B[1, 1], t[1]],
    ], dtype=np.float64)


def _points_to_dense_warp(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Interpolate sparse point correspondences into a dense (H, W, 2) pixel-space
    displacement field using the repo's `create_displacement_field`.
    """
    return create_displacement_field(
        source_pts,
        target_pts,
        image_shape,
        method=RegistrationParams.INTERPOLATION_METHOD,
        sigma=RegistrationParams.DISPLACEMENT_SIGMA,
        max_displacement=RegistrationParams.MAX_DISPLACEMENT,
    )


# ── 11. SIFT + RANSAC Affine ──────────────────────────────────────────────────

def run_sift_ransac(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    fixed_lm: Optional[np.ndarray],
    moving_lm: Optional[np.ndarray],
) -> MethodResult:
    """SIFT keypoint matching + RANSAC affine estimation."""
    result = MethodResult(method="sift_ransac", label=METHOD_LABELS["sift_ransac"])
    t0 = time.perf_counter()
    try:
        M = _keypoint_affine(source, target, descriptor="SIFT")
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


# ── 12. ICP ───────────────────────────────────────────────────────────────────

def run_icp(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    fixed_lm: Optional[np.ndarray],
    moving_lm: Optional[np.ndarray],
) -> MethodResult:
    """
    Iterative Closest Point (Open3D) on subsampled tissue mask points.
    The resulting rigid transform is applied to warp the full image.
    """
    result = MethodResult(method="icp", label=METHOD_LABELS["icp"])
    t0 = time.perf_counter()
    try:
        src_pts = _extract_tissue_points(source_mask)
        tgt_pts = _extract_tissue_points(target_mask)

        transform_4x4, _ = perform_icp_registration(src_pts, tgt_pts)

        # Convert 4×4 → 3×3 → 2×3 for cv2.warpAffine
        M3 = convert_4x4_to_3x3_transform(transform_4x4)
        M = M3[:2]   # 2×3

        warped = cv2.warpAffine(source, M, (target.shape[1], target.shape[0]))
        result.time_total_s = time.perf_counter() - t0

        warped_mask = _warp_mask(source_mask, M)
        _compute_image_metrics(target, warped, target_mask, warped_mask, result)

        if fixed_lm is not None and moving_lm is not None:
            warped_lm = _warp_landmarks_affine(moving_lm, np.linalg.inv(M3))
            _compute_landmark_metrics(fixed_lm, warped_lm, target.shape, result)
    except Exception as exc:
        result.note = str(exc)
    return result


# ── 13. CPD Rigid ─────────────────────────────────────────────────────────────

def run_cpd_rigid(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    fixed_lm: Optional[np.ndarray],
    moving_lm: Optional[np.ndarray],
) -> MethodResult:
    """
    Coherent Point Drift – Rigid variant on subsampled tissue mask points.
    Recovers rotation, uniform scale, and translation; applies them as an
    affine warp to the full image.
    """
    result = MethodResult(method="cpd_rigid", label=METHOD_LABELS["cpd_rigid"])
    t0 = time.perf_counter()
    try:
        src_pts = _extract_tissue_points(source_mask).astype(np.float64)
        tgt_pts = _extract_tissue_points(target_mask).astype(np.float64)

        cpd = _CPD(method="rigid")
        cpd(tgt_pts, src_pts, save_parameters=True)      # X=fixed, Y=moving
        R, t, s = cpd.parameters[-1]                      # final theta

        M = _cpd_rigid_to_2x3(R, t, s)
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


# ── 14. CPD Affine ────────────────────────────────────────────────────────────

def run_cpd_affine(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    fixed_lm: Optional[np.ndarray],
    moving_lm: Optional[np.ndarray],
) -> MethodResult:
    """
    Coherent Point Drift – Affine variant on subsampled tissue mask points.
    Allows independent scaling and shear on top of rotation/translation.
    """
    result = MethodResult(method="cpd_affine", label=METHOD_LABELS["cpd_affine"])
    t0 = time.perf_counter()
    try:
        src_pts = _extract_tissue_points(source_mask).astype(np.float64)
        tgt_pts = _extract_tissue_points(target_mask).astype(np.float64)

        cpd = _CPD(method="affine")
        cpd(tgt_pts, src_pts, save_parameters=True)      # X=fixed, Y=moving
        B, t = cpd.parameters[-1]

        M = _cpd_affine_to_2x3(B, t)
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


# ── 15. CPD Non-rigid ─────────────────────────────────────────────────────────

def run_cpd_nonrigid(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    fixed_lm: Optional[np.ndarray],
    moving_lm: Optional[np.ndarray],
) -> MethodResult:
    """
    Coherent Point Drift – Non-rigid (deformable) variant.
    Point correspondences are interpolated to a dense displacement field which
    is then applied to warp the full image.
    """
    result = MethodResult(method="cpd_nonrigid", label=METHOD_LABELS["cpd_nonrigid"])
    t0 = time.perf_counter()
    try:
        src_pts = _extract_tissue_points(source_mask).astype(np.float64)
        tgt_pts = _extract_tissue_points(target_mask).astype(np.float64)

        # Use repo's pycpd-backed CPD nonrigid (DeformableRegistration)
        ty, _ = perform_cpd_registration(src_pts, tgt_pts)

        # Build dense displacement field from the sparse correspondence src→ty
        disp_hw2 = _points_to_dense_warp(src_pts, ty, source.shape[:2])

        warped = util.apply_displacement_field(source, disp_hw2.transpose(2, 0, 1))
        result.time_total_s = time.perf_counter() - t0

        warped_mask = _warp_mask_displacement(source_mask, disp_hw2)
        _compute_image_metrics(target, warped, target_mask, warped_mask, result)

        if fixed_lm is not None and moving_lm is not None:
            warped_lm = _warp_landmarks_displacement(moving_lm, disp_hw2)
            _compute_landmark_metrics(fixed_lm, warped_lm, target.shape, result)
    except Exception as exc:
        result.note = str(exc)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# External-tool baselines: Elastix, NiftyReg, DROP
# ─────────────────────────────────────────────────────────────────────────────

# Elastix parameter-file templates
_ELASTIX_PARAM_AFFINE = """\
(Transform "AffineTransform")
(NumberOfHistogramBins 32)
(Metric "AdvancedMattesMutualInformation")
(NumberOfResolutions 3)
(MaximumNumberOfIterations 500)
(Interpolator "LinearInterpolator")
(ResampleInterpolator "FinalLinearInterpolator")
(Resampler "DefaultResampler")
(ImageSampler "RandomCoordinate")
(NumberOfSpatialSamples 2048)
(CheckNumberOfSamples "false")
(HowToCombineTransforms "Compose")
(AutomaticScalesEstimation "true")
(AutomaticTransformInitialization "true")
(WriteTransformParametersEachIteration "false")
(WriteResultImage "true")
(ResultImageFormat "mhd")
(ResultImagePixelType "float")
"""

_ELASTIX_PARAM_BSPLINE = """\
(Transform "BSplineTransform")
(NumberOfHistogramBins 32)
(Metric "AdvancedMattesMutualInformation")
(NumberOfResolutions 3)
(MaximumNumberOfIterations 500)
(Interpolator "LinearInterpolator")
(ResampleInterpolator "FinalLinearInterpolator")
(Resampler "DefaultResampler")
(ImageSampler "RandomCoordinate")
(NumberOfSpatialSamples 2048)
(CheckNumberOfSamples "false")
(HowToCombineTransforms "Compose")
(FinalGridSpacingInPhysicalUnits 16.0)
(GridSpacingSchedule 4.0 2.0 1.0)
(WriteTransformParametersEachIteration "false")
(WriteResultImage "true")
(ResultImageFormat "mhd")
(ResultImagePixelType "float")
"""


def _run_elastix(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    fixed_lm: Optional[np.ndarray],
    moving_lm: Optional[np.ndarray],
    param_content: str,
    method_key: str,
) -> MethodResult:
    """
    Generic Elastix runner.  Writes source/target to a temp directory, calls
    ``elastix``, reads back the warped result, and optionally calls
    ``transformix --def all`` to obtain a dense displacement field for
    landmark warping.

    Parameters
    ----------
    param_content : str
        Text content of the Elastix parameter file (affine or B-spline).
    method_key : str
        Key in ``ALL_METHODS`` used to populate ``MethodResult``.
    """
    result = MethodResult(method=method_key, label=METHOD_LABELS[method_key])
    t0 = time.perf_counter()
    tmp = tempfile.mkdtemp(prefix="elastix_")
    try:
        fixed_path  = os.path.join(tmp, "fixed.mhd")
        moving_path = os.path.join(tmp, "moving.mhd")
        param_path  = os.path.join(tmp, "param.txt")
        out_dir     = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)

        _save_mhd(target, fixed_path)
        _save_mhd(source, moving_path)
        with open(param_path, "w") as fh:
            fh.write(param_content)

        rc, stdout, stderr = _run_cmd([
            "elastix",
            "-f", fixed_path,
            "-m", moving_path,
            "-out", out_dir,
            "-p", param_path,
        ])
        if rc != 0:
            raise RuntimeError(
                f"elastix exited with code {rc}.\nstderr: {stderr.strip()}"
            )

        result_img_path = os.path.join(out_dir, "result.0.mhd")
        warped = _load_warped_mhd(result_img_path)
        result.time_total_s = time.perf_counter() - t0

        # Warp mask via affine approximation (cheapest fallback)
        warped_mask = source_mask.copy()   # default: no change

        # Optional: run transformix to get dense displacement field
        tp_path = os.path.join(out_dir, "TransformParameters.0.txt")
        if os.path.isfile(tp_path):
            tf_out = os.path.join(tmp, "tf_out")
            os.makedirs(tf_out, exist_ok=True)
            rc2, _, stderr2 = _run_cmd([
                "transformix",
                "-in", moving_path,
                "-out", tf_out,
                "-def", "all",
                "-tp", tp_path,
            ])
            def_path = os.path.join(tf_out, "deformationField.mhd")
            if rc2 == 0 and os.path.isfile(def_path):
                disp_hw2 = _load_mhd_displacement(def_path)
                warped_mask = _warp_mask_displacement(source_mask, disp_hw2)
                if fixed_lm is not None and moving_lm is not None:
                    warped_lm = _warp_landmarks_displacement(moving_lm, disp_hw2)
                    _compute_landmark_metrics(fixed_lm, warped_lm, target.shape, result)

        _compute_image_metrics(target, warped, target_mask, warped_mask, result)
    except Exception as exc:
        result.note = str(exc)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return result


# ── 16. Elastix Affine ────────────────────────────────────────────────────────

def run_elastix_affine(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    fixed_lm: Optional[np.ndarray],
    moving_lm: Optional[np.ndarray],
) -> MethodResult:
    """Elastix affine registration (AdvancedMattesMutualInformation metric)."""
    return _run_elastix(
        source, target, source_mask, target_mask, fixed_lm, moving_lm,
        param_content=_ELASTIX_PARAM_AFFINE,
        method_key="elastix_affine",
    )


# ── 17. Elastix B-Spline ──────────────────────────────────────────────────────

def run_elastix_bspline(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    fixed_lm: Optional[np.ndarray],
    moving_lm: Optional[np.ndarray],
) -> MethodResult:
    """Elastix B-spline non-rigid registration (FinalGridSpacing = 16 px)."""
    return _run_elastix(
        source, target, source_mask, target_mask, fixed_lm, moving_lm,
        param_content=_ELASTIX_PARAM_BSPLINE,
        method_key="elastix_bspline",
    )


# ── 18. NiftyReg Affine (reg_aladin) ─────────────────────────────────────────

def run_niftyreg_affine(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    fixed_lm: Optional[np.ndarray],
    moving_lm: Optional[np.ndarray],
) -> MethodResult:
    """
    NiftyReg rigid/affine registration using ``reg_aladin``.

    The output affine matrix (4×4 text file) is used both to warp the image
    and to propagate landmarks when provided.
    """
    result = MethodResult(method="niftyreg_affine", label=METHOD_LABELS["niftyreg_affine"])
    t0 = time.perf_counter()
    tmp = tempfile.mkdtemp(prefix="niftyreg_aff_")
    try:
        ref_path    = os.path.join(tmp, "ref.mhd")
        flo_path    = os.path.join(tmp, "flo.mhd")
        res_path    = os.path.join(tmp, "res.mhd")
        aff_path    = os.path.join(tmp, "affine.txt")

        _save_mhd(target, ref_path)
        _save_mhd(source, flo_path)

        rc, stdout, stderr = _run_cmd([
            "reg_aladin",
            "-ref", ref_path,
            "-flo", flo_path,
            "-res", res_path,
            "-aff", aff_path,
        ])
        if rc != 0:
            raise RuntimeError(
                f"reg_aladin exited with code {rc}.\nstderr: {stderr.strip()}"
            )

        warped = _load_warped_mhd(res_path)
        result.time_total_s = time.perf_counter() - t0

        # Build 2×3 affine matrix for mask / landmark warping
        if os.path.isfile(aff_path):
            M = _niftyreg_affine_to_2x3(aff_path)
            warped_mask = _warp_mask(source_mask, M)
            if fixed_lm is not None and moving_lm is not None:
                M3 = np.vstack([M, [0, 0, 1]])
                warped_lm = _warp_landmarks_affine(moving_lm, np.linalg.inv(M3))
                _compute_landmark_metrics(fixed_lm, warped_lm, target.shape, result)
        else:
            warped_mask = source_mask.copy()

        _compute_image_metrics(target, warped, target_mask, warped_mask, result)
    except Exception as exc:
        result.note = str(exc)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return result


# ── 19. NiftyReg B-Spline (reg_f3d) ──────────────────────────────────────────

def run_niftyreg_f3d(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    fixed_lm: Optional[np.ndarray],
    moving_lm: Optional[np.ndarray],
) -> MethodResult:
    """
    NiftyReg free-form deformation using ``reg_f3d``.

    Initialised with an affine pre-registration (``reg_aladin``) for
    robustness.  Landmark metrics are skipped because extracting a
    dense displacement field from the control-point grid would require
    ``reg_resample`` (available if NiftyReg is installed but not mandatory
    for the image-quality subset of metrics).
    """
    result = MethodResult(method="niftyreg_f3d", label=METHOD_LABELS["niftyreg_f3d"])
    t0 = time.perf_counter()
    tmp = tempfile.mkdtemp(prefix="niftyreg_f3d_")
    try:
        ref_path = os.path.join(tmp, "ref.mhd")
        flo_path = os.path.join(tmp, "flo.mhd")
        aff_path = os.path.join(tmp, "init_affine.txt")
        res_path = os.path.join(tmp, "res_f3d.mhd")
        cpp_path = os.path.join(tmp, "cpp.nii")

        _save_mhd(target, ref_path)
        _save_mhd(source, flo_path)

        # Step 1: affine initialisation
        rc, _, stderr = _run_cmd([
            "reg_aladin",
            "-ref", ref_path, "-flo", flo_path,
            "-res", os.path.join(tmp, "aff_res.mhd"),
            "-aff", aff_path,
        ])
        if rc != 0:
            raise RuntimeError(
                f"reg_aladin (init) exited with code {rc}.\nstderr: {stderr.strip()}"
            )

        # Step 2: non-rigid F3D
        cmd_f3d = [
            "reg_f3d",
            "-ref", ref_path, "-flo", flo_path,
            "-res", res_path,
            "-cpp", cpp_path,
        ]
        if os.path.isfile(aff_path):
            cmd_f3d += ["-aff", aff_path]

        rc2, _, stderr2 = _run_cmd(cmd_f3d)
        if rc2 != 0:
            raise RuntimeError(
                f"reg_f3d exited with code {rc2}.\nstderr: {stderr2.strip()}"
            )

        warped = _load_warped_mhd(res_path)
        result.time_total_s = time.perf_counter() - t0

        # Approximate mask warp with affine only (dense field not extracted here)
        if os.path.isfile(aff_path):
            M = _niftyreg_affine_to_2x3(aff_path)
            warped_mask = _warp_mask(source_mask, M)
        else:
            warped_mask = source_mask.copy()

        _compute_image_metrics(target, warped, target_mask, warped_mask, result)
        # Landmark metrics skipped (dense CPP→displacement conversion not performed)
        if fixed_lm is not None and moving_lm is not None:
            result.note = "Landmark metrics skipped for reg_f3d (no dense field extracted)."
    except Exception as exc:
        result.note = str(exc)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return result


# ── 20. DROP Deformable Registration ─────────────────────────────────────────

def run_drop(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    fixed_lm: Optional[np.ndarray],
    moving_lm: Optional[np.ndarray],
) -> MethodResult:
    """
    DROP (Deformable Registration using Discrete OPtimization).

    Calls the ``DROP2`` binary (alias ``DROP`` is tried as a fallback).
    DROP writes a dense MHD displacement field which is used to warp both
    the image and – when provided – landmark coordinates.

    Reference
    ---------
    Glocker et al., "Dense Image Registration through MRFs and Efficient
    Linear Programming." MedIA, 2008.
    """
    result = MethodResult(method="drop", label=METHOD_LABELS["drop"])
    t0 = time.perf_counter()
    tmp = tempfile.mkdtemp(prefix="drop_")
    try:
        fixed_path  = os.path.join(tmp, "fixed.mhd")
        moving_path = os.path.join(tmp, "moving.mhd")
        out_prefix  = os.path.join(tmp, "drop_result")

        _save_mhd(target, fixed_path)
        _save_mhd(source, moving_path)

        # Try DROP2 first, fall back to DROP
        last_error: Optional[str] = None
        success = False
        for exe in ("DROP2", "DROP"):
            try:
                rc, stdout, stderr = _run_cmd([
                    exe,
                    fixed_path,
                    moving_path,
                    out_prefix,
                ])
            except RuntimeError as exc:
                last_error = str(exc)
                continue
            if rc == 0:
                success = True
                break
            last_error = f"{exe} exited with code {rc}.\nstderr: {stderr.strip()}"

        if not success:
            raise RuntimeError(
                last_error or "Neither DROP2 nor DROP could be executed."
            )

        # DROP typically writes the warped image as <out_prefix>.mhd and the
        # displacement field as <out_prefix>_disp.mhd (or similar naming).
        warped_path = out_prefix + ".mhd"
        disp_path   = out_prefix + "_disp.mhd"

        if not os.path.isfile(warped_path):
            # Some DROP builds write the registered image with a different name
            candidates = [
                f for f in os.listdir(tmp)
                if f.endswith(".mhd") and "fixed" not in f and "moving" not in f
                and "disp" not in f
            ]
            if candidates:
                warped_path = os.path.join(tmp, candidates[0])
            else:
                raise RuntimeError(
                    "DROP finished but no warped image found in the output directory."
                )

        warped = _load_warped_mhd(warped_path)
        result.time_total_s = time.perf_counter() - t0

        warped_mask = source_mask.copy()
        if os.path.isfile(disp_path):
            disp_hw2 = _load_mhd_displacement(disp_path)
            warped_mask = _warp_mask_displacement(source_mask, disp_hw2)
            if fixed_lm is not None and moving_lm is not None:
                warped_lm = _warp_landmarks_displacement(moving_lm, disp_hw2)
                _compute_landmark_metrics(fixed_lm, warped_lm, target.shape, result)

        _compute_image_metrics(target, warped, target_mask, warped_mask, result)
    except Exception as exc:
        result.note = str(exc)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return result

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
    """Create and save all comparison plots for the ablation study."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available; skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    labels = [r.label for r in results]

    # ── 1. Image-quality metrics bar chart ────────────────────────────────
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

    # ── 2. Runtime bar chart ──────────────────────────────────────────────
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

    # ── 3. Landmark error bar chart (optional) ────────────────────────────
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

    # ── 4. Normalized metric heatmap ──────────────────────────────────────
    _heatmap_chart(
        results,
        path=os.path.join(output_dir, "ablation_heatmap.png"),
        has_landmarks=has_landmarks,
    )

    # ── 5. Quality vs. time scatter ───────────────────────────────────────
    _scatter_quality_vs_time(
        results,
        path=os.path.join(output_dir, "ablation_quality_vs_time.png"),
    )

    # ── 6. Per-metric ranked horizontal bars ──────────────────────────────
    _ranked_hbar_chart(
        results,
        path=os.path.join(output_dir, "ablation_ranked_metrics.png"),
        has_landmarks=has_landmarks,
    )

    # ── 7. Violin plots ───────────────────────────────────────────────────
    _violin_chart(
        results,
        path=os.path.join(output_dir, "ablation_violin.png"),
        has_landmarks=has_landmarks,
    )

    print(f"Plots saved to: {output_dir}")


_BAR_GROUP_OFFSET = 0.4   # centres the group of bars around each tick


def _violin_chart(
    results: List[MethodResult],
    path: str,
    has_landmarks: bool,
) -> None:
    """
    Violin plot – one violin per key metric, showing the distribution of
    values across all methods.

    Each violin is overlaid with:
    * a miniature box-plot (median bar + IQR box + whiskers)
    * individual method dots (jittered horizontally) labelled on hover
      – labels are added as annotations for the saved PNG instead.

    When fewer than 4 valid data points are available for a metric the
    violin is replaced by a plain box-plot so the chart is always useful.
    """
    import matplotlib.pyplot as plt

    metric_specs: List[Tuple[str, str, bool]] = [
        # (attr, display_name, lower_is_better)
        ("ncc",          "NCC ↑",            False),
        ("ssim_score",   "SSIM ↑",           False),
        ("ngf",          "NGF ↑",            False),
        ("dice_mask",    "DICE mask ↑",      False),
        ("time_total_s", "Time total (s) ↓", True),
    ]
    if has_landmarks:
        metric_specs += [
            ("tre_mean",  "TRE mean (px) ↓", True),
            ("rtre_mean", "rTRE mean ↓",      True),
        ]

    n_metrics = len(metric_specs)
    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(n_metrics * 2.8, 5),
        sharey=False,
    )
    if n_metrics == 1:
        axes = [axes]

    rng = np.random.default_rng(seed=42)  # reproducible jitter

    for ax, (attr, title, lower_is_better) in zip(axes, metric_specs):
        vals_all = np.array([
            getattr(r, attr) for r in results
            if not np.isnan(getattr(r, attr))
        ], dtype=float)
        labels_all = [
            r.label for r in results
            if not np.isnan(getattr(r, attr))
        ]

        if vals_all.size == 0:
            ax.set_title(title, fontsize=9)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.axis("off")
            continue

        color = "#e07070" if lower_is_better else "#70b0e0"

        if vals_all.size >= 4:
            # Violin with embedded mini box-plot
            vp = ax.violinplot(
                vals_all,
                positions=[1],
                widths=0.6,
                showmedians=False,
                showextrema=False,
            )
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.55)
                body.set_edgecolor("grey")
                body.set_linewidth(0.8)

            # Box-plot inset (no fliers – we draw our own dots)
            bp = ax.boxplot(
                vals_all,
                positions=[1],
                widths=0.12,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black", linewidth=1.5),
                boxprops=dict(facecolor="white", linewidth=0.8),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
            )
        else:
            # Fallback: plain box-plot when too few points for a violin
            bp = ax.boxplot(
                vals_all,
                positions=[1],
                widths=0.4,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black", linewidth=1.5),
                boxprops=dict(facecolor=color, alpha=0.55, linewidth=0.8),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
            )

        # Jittered dots for each method
        jitter = rng.uniform(-0.08, 0.08, size=vals_all.size)
        ax.scatter(
            np.ones(vals_all.size) + jitter,
            vals_all,
            s=30,
            color="steelblue",
            zorder=5,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.4,
        )

        # Annotate the best and worst dots (skip worst when identical)
        best_idx = int(np.argmin(vals_all) if lower_is_better else np.argmax(vals_all))
        worst_idx = int(np.argmax(vals_all) if lower_is_better else np.argmin(vals_all))
        annotations = [(best_idx, "best")]
        if worst_idx != best_idx:
            annotations.append((worst_idx, "worst"))
        for idx, tag in annotations:
            ax.annotate(
                labels_all[idx],
                xy=(1 + jitter[idx], vals_all[idx]),
                xytext=(8, 0),
                textcoords="offset points",
                fontsize=6,
                color="darkgreen" if tag == "best" else "firebrick",
                va="center",
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
            )

        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Distribution of Metric Values Across Methods\n"
        "(each dot = one method; best/worst labelled)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
        bars = ax.bar(x + i * width - _BAR_GROUP_OFFSET + width / 2, vals_clean, width, label=name)

    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _heatmap_chart(
    results: List[MethodResult],
    path: str,
    has_landmarks: bool,
) -> None:
    """
    Normalized metric heatmap (methods × metrics).

    Each column is independently min-max normalized to [0, 1] so that
    different scales are comparable side-by-side.  For "lower is better"
    metrics (TRE, rTRE, Time) the normalization is inverted so that the
    brightest cell always indicates the *best* performer.

    Cells with no data (NaN) are shown in grey.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Define which metrics to show and whether lower is better
    metric_defs: List[Tuple[str, bool]] = [
        ("NGF ↑",          False),   # higher = better → normalize as-is
        ("NCC ↑",          False),
        ("SSIM ↑",         False),
        ("DICE mask ↑",    False),
        ("Time total (s)", True),    # lower = better → invert
    ]
    if has_landmarks:
        metric_defs += [
            ("TRE mean (px) ↓",    True),
            ("TRE median (px) ↓",  True),
            ("rTRE mean ↓",        True),
        ]

    # Attribute names on MethodResult corresponding to each metric
    attr_map = {
        "NGF ↑":           "ngf",
        "NCC ↑":           "ncc",
        "SSIM ↑":          "ssim_score",
        "DICE mask ↑":     "dice_mask",
        "Time total (s)":  "time_total_s",
        "TRE mean (px) ↓":   "tre_mean",
        "TRE median (px) ↓": "tre_median",
        "rTRE mean ↓":       "rtre_mean",
    }

    labels = [r.label for r in results]
    n_methods = len(results)
    n_metrics = len(metric_defs)

    # Build raw matrix (n_methods × n_metrics)
    raw = np.full((n_methods, n_metrics), np.nan)
    for j, (col_name, _) in enumerate(metric_defs):
        attr = attr_map[col_name]
        for i, r in enumerate(results):
            raw[i, j] = getattr(r, attr)

    # Normalize each column independently
    norm = np.full_like(raw, np.nan)
    for j, (_, lower_is_better) in enumerate(metric_defs):
        col = raw[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            norm[:, j] = col
            continue
        lo, hi = col[valid].min(), col[valid].max()
        if hi == lo:
            norm[valid, j] = 1.0
        else:
            norm[valid, j] = (col[valid] - lo) / (hi - lo)
            if lower_is_better:
                norm[valid, j] = 1.0 - norm[valid, j]

    # Plot
    col_labels = [name for name, _ in metric_defs]
    fig, ax = plt.subplots(
        figsize=(max(8, n_metrics * 1.2), max(4, n_methods * 0.55))
    )

    cmap = plt.get_cmap("RdYlGn")
    cmap.set_bad(color="#cccccc")
    masked = np.ma.array(norm, mask=np.isnan(norm))
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(np.arange(n_metrics))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(np.arange(n_methods))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(
        "Method Comparison Heatmap\n(green = best, red = worst per column; grey = N/A)",
        fontsize=11,
    )

    # Annotate cells with raw values
    for i in range(n_methods):
        for j in range(n_metrics):
            v = raw[i, j]
            txt = "N/A" if np.isnan(v) else f"{v:.3f}"
            brightness = norm[i, j] if not np.isnan(norm[i, j]) else 0.5
            text_color = "black" if 0.25 < brightness < 0.85 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=text_color)

    plt.colorbar(im, ax=ax, label="Normalized score (1 = best)", fraction=0.02, pad=0.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _scatter_quality_vs_time(
    results: List[MethodResult],
    path: str,
) -> None:
    """
    Scatter plot of registration quality (NCC on y-axis, SSIM on secondary
    y-axis) versus total wall-clock time (log-scale x-axis).

    Each method is a labeled point.  Methods that failed (NaN time or NCC)
    are omitted from this plot.
    """
    import matplotlib.pyplot as plt

    valid = [r for r in results if not np.isnan(r.time_total_s) and not np.isnan(r.ncc)]
    if not valid:
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    times  = np.array([r.time_total_s for r in valid])
    nccs   = np.array([r.ncc          for r in valid])
    ssims  = np.array([r.ssim_score   for r in valid])

    # Use a categorical color map so each method gets a distinct colour.
    # Generate evenly-spaced hues via HSV when there are more than 20 methods.
    n_valid = len(valid)
    if n_valid <= 20:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(n_valid)]
    else:
        cmap = plt.get_cmap("hsv")
        colors = [cmap(i / n_valid) for i in range(n_valid)]

    for i, r in enumerate(valid):
        ax1.scatter(r.time_total_s, r.ncc, color=colors[i], s=80, zorder=3)
        if not np.isnan(r.ssim_score):
            ax2.scatter(r.time_total_s, r.ssim_score, color=colors[i],
                        marker="^", s=60, zorder=3, alpha=0.7)
        # Label offset to avoid overlapping the point
        ax1.annotate(
            r.label,
            (r.time_total_s, r.ncc),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=7,
            color=colors[i],
        )

    ax1.set_xscale("log")
    ax1.set_xlabel("Total registration time (s, log scale)", fontsize=10)
    ax1.set_ylabel("NCC ↑ (circles)", fontsize=10, color="black")
    ax2.set_ylabel("SSIM ↑ (triangles)", fontsize=10, color="dimgrey")
    ax1.set_title("Quality vs. Runtime Trade-off", fontsize=12)
    ax1.grid(True, which="both", alpha=0.3)

    # Legend: one entry per method
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[i],
                   markersize=8, label=r.label)
        for i, r in enumerate(valid)
    ]
    ax1.legend(handles=handles, fontsize=7, loc="lower right", ncol=2,
               framealpha=0.85)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _ranked_hbar_chart(
    results: List[MethodResult],
    path: str,
    has_landmarks: bool,
) -> None:
    """
    One horizontal-bar subplot per key metric, methods sorted best → worst.

    This makes it easy to see which method wins on each individual criterion.
    """
    import matplotlib.pyplot as plt

    metric_specs: List[Tuple[str, str, bool]] = [
        # (attr, display_name, lower_is_better)
        ("ncc",        "NCC ↑",          False),
        ("ssim_score", "SSIM ↑",         False),
        ("ngf",        "NGF ↑",          False),
        ("dice_mask",  "DICE mask ↑",    False),
        ("time_total_s", "Time total (s) ↓", True),
    ]
    if has_landmarks:
        metric_specs += [
            ("tre_mean",   "TRE mean (px) ↓",  True),
            ("rtre_mean",  "rTRE mean ↓",       True),
        ]

    n_plots = len(metric_specs)
    fig, axes = plt.subplots(
        1, n_plots,
        figsize=(n_plots * 3.5, max(4, len(results) * 0.45)),
        sharey=False,
    )
    if n_plots == 1:
        axes = [axes]

    cmap = plt.get_cmap("RdYlGn")

    for ax, (attr, title, lower_is_better) in zip(axes, metric_specs):
        # Filter to methods with valid data
        pairs = [
            (getattr(r, attr), r.label)
            for r in results
            if not np.isnan(getattr(r, attr))
        ]
        if not pairs:
            ax.set_title(title, fontsize=9)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.axis("off")
            continue

        # Sort: ascending for lower-is-better, descending otherwise
        pairs.sort(key=lambda x: x[0], reverse=not lower_is_better)
        vals, lbls = zip(*pairs)
        vals = np.array(vals, dtype=float)

        # Color bars: normalize within the visible range
        lo, hi = vals.min(), vals.max()
        if hi > lo:
            norm_vals = (vals - lo) / (hi - lo)
        else:
            norm_vals = np.ones_like(vals)
        # For lower-is-better invert so green = smallest
        if lower_is_better:
            bar_colors = [cmap(1.0 - v) for v in norm_vals]
        else:
            bar_colors = [cmap(v) for v in norm_vals]

        y = np.arange(len(vals))
        ax.barh(y, vals, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(lbls, fontsize=7)
        ax.invert_yaxis()   # best at the top
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Value", fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", alpha=0.35)

    fig.suptitle("Per-Metric Method Ranking (best at top)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
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
        # ── Point-based methods ──────────────────────────────────────────
        "sift_ransac":       lambda: run_sift_ransac(**shared_kwargs),
        "icp":               lambda: run_icp(**shared_kwargs),
        "cpd_rigid":         lambda: run_cpd_rigid(**shared_kwargs),
        "cpd_affine":        lambda: run_cpd_affine(**shared_kwargs),
        "cpd_nonrigid":      lambda: run_cpd_nonrigid(**shared_kwargs),
        # ── External registration toolboxes ─────────────────────────────
        "elastix_affine":    lambda: run_elastix_affine(**shared_kwargs),
        "elastix_bspline":   lambda: run_elastix_bspline(**shared_kwargs),
        "niftyreg_affine":   lambda: run_niftyreg_affine(**shared_kwargs),
        "niftyreg_f3d":      lambda: run_niftyreg_f3d(**shared_kwargs),
        "drop":              lambda: run_drop(**shared_kwargs),
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
            "  CORE ablation : baseline rigid_only elastic_only core_coarse core_full\n"
            "  SOTA          : phase_correlation itk_rigid_mi itk_demons\n"
            "                  orb_ransac akaze_ransac sift_ransac\n"
            "  Point-based   : icp cpd_rigid cpd_affine cpd_nonrigid\n"
            "  External tools: elastix_affine elastix_bspline\n"
            "                  niftyreg_affine niftyreg_f3d drop\n"
            "  (External tools require their respective binaries on PATH;\n"
            "   if missing, the method is skipped with an informative note.)"
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
