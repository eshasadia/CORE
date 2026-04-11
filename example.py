"""
CORE – Example Script
=====================
Demonstrates the full coarse-to-fine WSI registration pipeline:

  1. Load source (moving) and target (fixed) whole-slide images.
  2. Pad images to a common canvas size.
  3. Extract tissue masks.
  4. Coarse rigid registration (XFeat-based).
  5. Coarse non-rigid / elastic registration.
  6. Save the combined deformation field as an MHA file.
  7. (Optional) Fine nuclei-level shape-aware registration.
  8. Visualise results with matplotlib overlays.

Before running
--------------
1.  Install dependencies::

        conda env create -f environment.yml
        conda activate core

2.  Set your VisionAgent API key (needed for tissue-mask extraction)::

        export VISION_AGENT_API_KEY="<your-key>"

3.  Update the path variables in the ``# ── User configuration ──`` section
    below to point at your own slides.
"""

import os
import sys

# ── Make sure the repo root is on sys.path when the script is run directly ──
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

from core.config import (
    PREPROCESSING_RESOLUTION,
    REGISTRATION_RESOLUTION,
    PATCH_SIZE,
    PATCH_STRIDE,
    VISUALIZATION_SIZE,
    FIXED_THRESHOLD,
    MOVING_THRESHOLD,
    MIN_NUCLEI_AREA,
    RegistrationParams,
)
from core.preprocessing.preprocessing import load_wsi_images, extract_tissue_masks
from core.preprocessing.padding import pad_images
from core.registration.registration import perform_rigid_registration
from core.registration.nonrigid import elastic_image_registration
import core.utils.util as util


# ── User configuration ──────────────────────────────────────────────────────
SOURCE_WSI_PATH = "/path/to/source.tiff"   # moving slide
TARGET_WSI_PATH = "/path/to/target.tiff"   # fixed slide

# Output path for the combined deformation field (MHA format).
DEFORMATION_OUTPUT_PATH = "./deformation_field.mha"

# Set to True to run the optional fine nuclei-level registration step.
RUN_FINE_REGISTRATION = False

# CSV files produced by a nuclei-detection pipeline (required only when
# RUN_FINE_REGISTRATION is True).
FIXED_NUCLEI_CSV  = "/path/to/fixed_nuclei.csv"
MOVING_NUCLEI_CSV = "/path/to/moving_nuclei.csv"
# ────────────────────────────────────────────────────────────────────────────


def load_and_preprocess():
    """Load WSIs, pad to a common canvas, and extract tissue masks."""
    print("── Step 1 · Loading WSI images ──────────────────────────────")
    source_wsi, target_wsi, source, target = load_wsi_images(
        SOURCE_WSI_PATH, TARGET_WSI_PATH, PREPROCESSING_RESOLUTION
    )
    print(f"  Source shape : {source.shape}")
    print(f"  Target shape : {target.shape}")

    print("\n── Step 2 · Padding images ──────────────────────────────────")
    source_prep, target_prep, padding_params = pad_images(source, target)
    print(f"  Padded source : {source_prep.shape}")
    print(f"  Padded target : {target_prep.shape}")

    print("\n── Step 3 · Extracting tissue masks ─────────────────────────")
    # Set artefacts=True if the slide contains control tissue regions.
    source_mask, target_mask = extract_tissue_masks(
        source_prep, target_prep, artefacts=False
    )
    print("  Tissue masks extracted ✓")

    return source_wsi, target_wsi, source_prep, target_prep, source_mask, target_mask, padding_params


def visualise_preprocessing(source_prep, target_prep, source_mask, target_mask):
    """Display the padded images and their tissue masks side-by-side."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(source_prep)
    axes[0, 0].set_title("Source (Moving)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(target_prep)
    axes[0, 1].set_title("Target (Fixed)")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(source_mask, cmap="gray")
    axes[1, 0].set_title("Source Tissue Mask")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(target_mask, cmap="gray")
    axes[1, 1].set_title("Target Tissue Mask")
    axes[1, 1].axis("off")

    plt.suptitle("Preprocessed Images and Tissue Masks", fontsize=14)
    plt.tight_layout()
    plt.show()


def coarse_registration(source_prep, target_prep, source_mask, target_mask):
    """Rigid + elastic coarse-level registration."""
    print("\n── Step 4 · Rigid (coarse) registration ─────────────────────")
    moving_img_transformed, final_transform = perform_rigid_registration(
        source_prep, target_prep, source_mask, target_mask
    )
    print("  Rigid registration complete ✓")

    # Quick visual check after rigid alignment
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(target_prep);            axes[0].set_title("Target (Fixed)");      axes[0].axis("off")
    axes[1].imshow(source_prep);            axes[1].set_title("Source (Moving)");     axes[1].axis("off")
    axes[2].imshow(moving_img_transformed); axes[2].set_title("After Rigid Reg.");    axes[2].axis("off")
    plt.suptitle("Coarse Rigid Registration Result", fontsize=14)
    plt.tight_layout()
    plt.show()

    print("\n── Step 5 · Elastic (non-rigid) registration ────────────────")
    displacement_field, warped_source = elastic_image_registration(
        moving_img_transformed, target_prep
    )
    print(f"  Displacement field shape : {displacement_field.shape}")
    print("  Elastic registration complete ✓")

    return moving_img_transformed, final_transform, displacement_field, warped_source


def save_deformation_field(source_prep, final_transform, displacement_field, output_path):
    """Combine rigid + non-rigid transforms into a single MHA deformation field."""
    print("\n── Step 6 · Saving combined deformation field ───────────────")
    sitk_image = util.create_deform(
        source_prep, final_transform, displacement_field, output_path=output_path
    )
    print(f"  Deformation field saved to: {output_path} ✓")
    return sitk_image


def fine_registration(source_prep, target_prep, final_transform, displacement_field, padding_params):
    """
    Optional nuclei-level shape-aware fine registration.

    Requires precomputed nuclei CSV files (global_x, global_y columns).
    """
    from core.preprocessing.nuclei_analysis import load_nuclei_coordinates
    from core.registration.registration import perform_shape_aware_registration
    from core.registration.nonrigid import compute_deformation_and_apply
    from core.preprocessing.padding import pad_landmarks

    print("\n── Step 7 · Fine nuclei-level registration ──────────────────")
    moving_df = load_nuclei_coordinates(MOVING_NUCLEI_CSV)
    fixed_df  = load_nuclei_coordinates(FIXED_NUCLEI_CSV)
    print(f"  Fixed nuclei : {len(fixed_df)}   Moving nuclei : {len(moving_df)}")

    # Apply the coarse deformation field to landmark coordinates
    deformation_field, moving_updated, fixed_points, moving_points = compute_deformation_and_apply(
        source_prep,
        final_transform,
        displacement_field,
        moving_df,
        fixed_df,
        padding_params,
        util,
        pad_landmarks,
    )

    # Shape-aware point-set registration on transformed nuclei
    print("  Running shape-aware point-set registration...")
    _, shape_transform, shape_transformed_coords = perform_shape_aware_registration(
        fixed_points,
        moving_updated,
        shape_weight=0.3,
        max_iterations=100,
        tolerance=1e-11,
    )
    print("  Fine registration complete ✓")
    return fixed_points, moving_updated, shape_transformed_coords


def visualise_overlay(target_prep, source_prep, registered):
    """Checkerboard-style overlay of fixed and registered images."""
    h, w = target_prep.shape[:2]
    tile = 64
    overlay = np.zeros_like(target_prep)
    for row in range(0, h, tile):
        for col in range(0, w, tile):
            use_fixed = ((row // tile) + (col // tile)) % 2 == 0
            src = target_prep if use_fixed else registered
            overlay[row:row+tile, col:col+tile] = src[row:row+tile, col:col+tile]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(source_prep);  axes[0].set_title("Before Registration"); axes[0].axis("off")
    axes[1].imshow(overlay);      axes[1].set_title("After Registration (Checkerboard)"); axes[1].axis("off")
    plt.suptitle("Registration Quality", fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    # ── 1–3: Load, pad, mask ──────────────────────────────────────────────
    (
        source_wsi, target_wsi,
        source_prep, target_prep,
        source_mask, target_mask,
        padding_params,
    ) = load_and_preprocess()

    visualise_preprocessing(source_prep, target_prep, source_mask, target_mask)

    # ── 4–5: Coarse registration ──────────────────────────────────────────
    moving_img_transformed, final_transform, displacement_field, warped_source = coarse_registration(
        source_prep, target_prep, source_mask, target_mask
    )

    # ── 6: Save combined deformation field ───────────────────────────────
    save_deformation_field(
        source_prep, final_transform, displacement_field, DEFORMATION_OUTPUT_PATH
    )

    # ── 7 (optional): Fine registration ──────────────────────────────────
    if RUN_FINE_REGISTRATION:
        fine_registration(
            source_prep, target_prep,
            final_transform, displacement_field,
            padding_params,
        )

    # ── 8: Final visualisation ────────────────────────────────────────────
    warped_np = warped_source.detach().cpu().numpy()
    if warped_np.ndim == 4:
        warped_np = warped_np[0, 0]          # (1,1,H,W) → (H,W)
    warped_rgb = np.stack([warped_np] * 3, axis=-1)
    warped_rgb = (warped_rgb * 255).clip(0, 255).astype(np.uint8)
    visualise_overlay(target_prep, source_prep, warped_rgb)

    print("\n✅ CORE registration pipeline complete.")
    print(f"   Deformation field : {DEFORMATION_OUTPUT_PATH}")
    print(
        "\nTo visualise deformation in TIAViz, run:\n"
        "  tiatoolbox visualize --slides <slides-folder> --overlays <overlays-folder>\n"
    )


if __name__ == "__main__":
    main()
