"""
batch_run.py
============
Run the CORE coarse-to-fine WSI registration pipeline on a batch of
slide pairs described by a CSV file.

Supported input formats
-----------------------
Both ``source_path`` and ``target_path`` may be any format supported by
tiatoolbox (WSIReader) and pyvips.  Recognised extensions include:

    Pyramidal WSI:  .svs, .ndpi, .mrxs, .scn, .vms, .vmu, .bif, .qptiff
    TIFF family:    .tiff, .tif, .ome.tiff, .ome.tif, .btf
    Other imaging:  .czi, .lif, .png, .jpg, .jpeg

The registered output is always written as a pyramidal OME-TIFF
(``.ome.tiff``), regardless of the input format.

CSV format
----------
The input CSV must contain at least two columns:

    source_path,target_path

Optional columns (override the CLI defaults for individual pairs):

    deformation_output   – path for the combined MHA deformation field
    wsi_output           – path for the registered OME-TIFF
    source_magnification – magnification at which the field was computed
    target_magnification – full-resolution magnification of the WSI
    fixed_nuclei_csv     – nuclei CSV for the fixed image  (fine-reg only)
    moving_nuclei_csv    – nuclei CSV for the moving image (fine-reg only)

Any column not present in the CSV row will fall back to the
corresponding CLI argument.

Usage
-----
    python batch_run.py \\
        --csv  pairs.csv \\
        --output-dir ./batch_results \\
        [--source-mag  0.625] \\
        [--target-mag  40.0] \\
        [--tile-size   512] \\
        [--compression lzw] \\
        [--fine-registration] \\
        [--no-visualise]

Before running
--------------
1.  Install dependencies:

        conda env create -f environment.yml
        conda activate core

2.  Set your VisionAgent API key:

        export VISION_AGENT_API_KEY="<your-key>"

3.  Prepare a CSV with at least ``source_path`` and ``target_path`` columns.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Make sure the repo root is on sys.path when run directly ──────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
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
from apply_deformation_wsi import apply_deformation_to_wsi


# ---------------------------------------------------------------------------
# Format support
# ---------------------------------------------------------------------------

# Extensions recognised by tiatoolbox (WSIReader / VirtualWSIReader) and pyvips.
# The check is case-insensitive; dual-suffix extensions (.ome.tiff) are listed
# explicitly so they are matched before the single-suffix fallback.
SUPPORTED_WSI_EXTENSIONS: tuple[str, ...] = (
    # Dual-suffix OME-TIFF (must come before single .tiff/.tif)
    ".ome.tiff",
    ".ome.tif",
    # Pyramidal WSI scanners
    ".svs",    # Aperio / Leica
    ".ndpi",   # Hamamatsu
    ".mrxs",   # 3DHISTECH Mirax
    ".scn",    # Leica SCN
    ".vms",    # Hamamatsu VMS
    ".vmu",    # Hamamatsu VMU
    ".bif",    # Ventana BIF
    ".qptiff", # PerkinElmer / Akoya
    # TIFF family
    ".tiff",
    ".tif",
    ".btf",    # Big TIFF
    # Other supported imaging formats
    ".czi",    # Zeiss CZI
    ".lif",    # Leica LIF
    ".png",
    ".jpg",
    ".jpeg",
)


def _wsi_stem(path: str) -> str:
    """Return the base filename without any WSI-related suffix.

    Handles dual-suffix filenames such as ``slide.ome.tiff`` correctly
    (``Path.stem`` alone would return ``slide.ome`` in that case).

    Examples
    --------
    >>> _wsi_stem("case1.svs")
    'case1'
    >>> _wsi_stem("/data/slide.ome.tiff")
    'slide'
    >>> _wsi_stem("my_slide.tif")
    'my_slide'
    """
    name = Path(path).name.lower()
    for ext in SUPPORTED_WSI_EXTENSIONS:
        if name.endswith(ext):
            # Preserve the original-case stem
            return Path(path).name[: -len(ext)]
    # Fallback: strip the last suffix
    return Path(path).stem


def _validate_wsi_path(path: str, label: str) -> None:
    """Raise ``ValueError`` if *path* has an unrecognised WSI extension.

    Parameters
    ----------
    path:
        Filesystem path to the WSI file.
    label:
        Human-readable label used in the error message (e.g. ``"source"``).
    """
    name = Path(path).name.lower()
    supported = any(name.endswith(ext) for ext in SUPPORTED_WSI_EXTENSIONS)
    if not supported:
        suffix = Path(path).suffix or "(no extension)"
        raise ValueError(
            f"Unsupported {label} WSI format: {suffix!r} ({path!r}).\n"
            f"Supported extensions: {', '.join(SUPPORTED_WSI_EXTENSIONS)}"
        )


# ---------------------------------------------------------------------------
# Per-pair registration logic
# ---------------------------------------------------------------------------

def _run_pair(
    source_path: str,
    target_path: str,
    deformation_output: str,
    wsi_output: str,
    source_magnification: float,
    target_magnification: float,
    tile_size: int,
    compression: str,
    run_fine_registration: bool,
    fixed_nuclei_csv: Optional[str],
    moving_nuclei_csv: Optional[str],
    visualise: bool,
) -> None:
    """Execute the full CORE pipeline for a single source/target pair."""

    # ── 1–3: Load, pad, mask ─────────────────────────────────────────────────
    print("  ── Step 1 · Loading WSI images ──")
    source_wsi, target_wsi, source, target = load_wsi_images(
        source_path, target_path, PREPROCESSING_RESOLUTION
    )
    print(f"    Source shape : {source.shape}")
    print(f"    Target shape : {target.shape}")

    print("  ── Step 2 · Padding images ──")
    source_prep, target_prep, padding_params = pad_images(source, target)
    print(f"    Padded source : {source_prep.shape}")
    print(f"    Padded target : {target_prep.shape}")

    print("  ── Step 3 · Extracting tissue masks ──")
    source_mask, target_mask = extract_tissue_masks(
        source_prep, target_prep, artefacts=False
    )
    print("    Tissue masks extracted ✓")

    # ── 4–5: Coarse registration ──────────────────────────────────────────────
    print("  ── Step 4 · Rigid (coarse) registration ──")
    moving_img_transformed, final_transform = perform_rigid_registration(
        source_prep, target_prep, source_mask, target_mask
    )
    print("    Rigid registration complete ✓")

    print("  ── Step 5 · Elastic (non-rigid) registration ──")
    displacement_field, warped_source = elastic_image_registration(
        moving_img_transformed, target_prep
    )
    print(f"    Displacement field shape : {displacement_field.shape}")
    print("    Elastic registration complete ✓")

    # ── 6: Save combined deformation field ───────────────────────────────────
    print(f"  ── Step 6 · Saving deformation field → {deformation_output} ──")
    Path(deformation_output).parent.mkdir(parents=True, exist_ok=True)
    util.create_deform(
        source_prep, final_transform, displacement_field,
        output_path=deformation_output,
    )
    print("    Deformation field saved ✓")

    # ── 7: Apply deformation to full-resolution WSI ───────────────────────────
    print(f"  ── Step 7 · Applying deformation to WSI → {wsi_output} ──")
    apply_deformation_to_wsi(
        mha_path=deformation_output,
        wsi_path=source_path,
        output_path=wsi_output,
        source_magnification=source_magnification,
        target_magnification=target_magnification,
        tile_size=tile_size,
        compression=compression,
    )
    print("    Registered WSI saved ✓")

    # ── 8 (optional): Fine nuclei-level registration ──────────────────────────
    if run_fine_registration:
        if not fixed_nuclei_csv or not moving_nuclei_csv:
            print("  ⚠  Fine registration skipped: nuclei CSVs not provided for this pair.")
        else:
            print("  ── Step 8 · Fine nuclei-level registration ──")
            from core.preprocessing.nuclei_analysis import load_nuclei_coordinates
            from core.registration.registration import perform_shape_aware_registration
            from core.registration.nonrigid import compute_deformation_and_apply
            from core.preprocessing.padding import pad_landmarks

            moving_df = load_nuclei_coordinates(moving_nuclei_csv)
            fixed_df  = load_nuclei_coordinates(fixed_nuclei_csv)
            print(f"    Fixed nuclei : {len(fixed_df)}   Moving nuclei : {len(moving_df)}")

            _, moving_updated, fixed_points, moving_points = compute_deformation_and_apply(
                source_prep, final_transform, displacement_field,
                moving_df, fixed_df, padding_params, util, pad_landmarks,
            )
            perform_shape_aware_registration(
                fixed_points, moving_updated,
                shape_weight=0.3, max_iterations=100, tolerance=1e-11,
            )
            print("    Fine registration complete ✓")

    # ── 9: Optional visualisation ─────────────────────────────────────────────
    if visualise:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend for batch use
        import matplotlib.pyplot as plt

        warped_np = warped_source.detach().cpu().numpy()
        if warped_np.ndim == 4:
            warped_np = warped_np[0, 0]
        warped_rgb = np.stack([warped_np] * 3, axis=-1)
        warped_rgb = (warped_rgb * 255).clip(0, 255).astype(np.uint8)

        vis_path = str(Path(wsi_output).with_suffix("")) + "_overlay.png"
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(source_prep); axes[0].set_title("Before Registration"); axes[0].axis("off")

        h, w = target_prep.shape[:2]
        tile = 64
        overlay = np.zeros_like(target_prep)
        for row in range(0, h, tile):
            for col in range(0, w, tile):
                use_fixed = ((row // tile) + (col // tile)) % 2 == 0
                src = target_prep if use_fixed else warped_rgb
                overlay[row:row+tile, col:col+tile] = src[row:row+tile, col:col+tile]

        axes[1].imshow(overlay); axes[1].set_title("After Registration (Checkerboard)"); axes[1].axis("off")
        plt.suptitle("Registration Quality", fontsize=14)
        plt.tight_layout()
        plt.savefig(vis_path, dpi=100)
        plt.close(fig)
        print(f"    Overlay saved → {vis_path}")


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def run_batch(
    csv_path: str,
    output_dir: str,
    source_magnification: float,
    target_magnification: float,
    tile_size: int,
    compression: str,
    run_fine_registration: bool,
    visualise: bool,
) -> None:
    """Process all pairs listed in *csv_path*."""
    df = pd.read_csv(csv_path)
    required = {"source_path", "target_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required column(s): {missing}\n"
            "The CSV must contain at least 'source_path' and 'target_path' columns."
        )

    n_pairs      = len(df)
    n_ok         = 0
    n_fail       = 0
    summary_rows = []

    print(f"\n{'=' * 60}")
    print(f"  CORE Batch Registration — {n_pairs} pair(s)")
    print(f"  Output directory : {output_dir}")
    print(f"{'=' * 60}\n")

    for pair_num, row in enumerate(df.itertuples(index=False), start=1):
        source_path = str(row.source_path).strip()
        target_path = str(row.target_path).strip()

        pair_label = f"Pair {pair_num}/{n_pairs}: {Path(source_path).name} ↔ {Path(target_path).name}"
        print(f"\n{'─' * 60}")
        print(f"  {pair_label}")
        print(f"{'─' * 60}")

        # Build per-pair output paths (use CSV columns if provided)
        pair_dir = Path(output_dir) / f"pair_{pair_num:04d}"
        stem = _wsi_stem(source_path)

        row_dict = row._asdict()

        def _get(name, default):
            val = row_dict.get(name)
            if val is not None and not (isinstance(val, float) and np.isnan(val)) and str(val).strip():
                return type(default)(val) if default is not None else str(val)
            return default

        deformation_output = (
            _get("deformation_output", None)
            or str(pair_dir / f"{stem}_deformation_field.mha")
        )
        wsi_output = (
            _get("wsi_output", None)
            or str(pair_dir / f"{stem}_registered.ome.tiff")
        )
        src_mag           = _get("source_magnification", source_magnification)
        tgt_mag           = _get("target_magnification",  target_magnification)
        fixed_nuclei_csv  = _get("fixed_nuclei_csv",  None)
        moving_nuclei_csv = _get("moving_nuclei_csv", None)

        status    = "OK"
        error_msg = ""
        try:
            # Validate input formats before running the pipeline
            _validate_wsi_path(source_path, "source")
            _validate_wsi_path(target_path, "target")

            _run_pair(
                source_path=source_path,
                target_path=target_path,
                deformation_output=deformation_output,
                wsi_output=wsi_output,
                source_magnification=src_mag,
                target_magnification=tgt_mag,
                tile_size=tile_size,
                compression=compression,
                run_fine_registration=run_fine_registration,
                fixed_nuclei_csv=fixed_nuclei_csv,
                moving_nuclei_csv=moving_nuclei_csv,
                visualise=visualise,
            )
            print(f"\n  ✅ {pair_label} — complete")
            n_ok += 1

        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            status    = "FAILED"
            error_msg = str(exc)
            print(f"\n  ❌ {pair_label} — FAILED: {exc}")
            traceback.print_exc()

        summary_rows.append({
            "pair":        pair_num,
            "source_path": source_path,
            "target_path": target_path,
            "status":      status,
            "error":       error_msg,
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Batch complete — {n_ok} succeeded, {n_fail} failed out of {n_pairs}")
    failures = [(r["pair"], r["source_path"], r["target_path"], r["error"])
                for r in summary_rows if r["status"] == "FAILED"]
    if failures:
        print("\n  Failed pairs:")
        for pair_num, src, tgt, err in failures:
            print(f"    [{pair_num}] {src} ↔ {tgt}")
            print(f"         Error: {err}")

    # Write summary CSV to the output directory
    summary_path = Path(output_dir) / "batch_summary.csv"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\n  Summary written → {summary_path}")
    print(f"{'=' * 60}\n")

    if n_fail:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CORE coarse-to-fine WSI registration pipeline on a "
            "batch of slide pairs described by a CSV file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--csv", "-c",
        required=True,
        metavar="PAIRS_CSV",
        help=(
            "Path to a CSV file with at least 'source_path' and 'target_path' "
            "columns.  Optional columns: deformation_output, wsi_output, "
            "source_magnification, target_magnification, fixed_nuclei_csv, "
            "moving_nuclei_csv."
        ),
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./batch_results",
        metavar="DIR",
        help="Root output directory; per-pair sub-directories are created automatically.",
    )
    parser.add_argument(
        "--source-mag",
        type=float,
        default=0.625,
        metavar="FLOAT",
        help="Magnification at which the deformation field was computed.",
    )
    parser.add_argument(
        "--target-mag",
        type=float,
        default=40.0,
        metavar="FLOAT",
        help="Full-resolution magnification of the WSI.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        metavar="INT",
        help="OME-TIFF output tile size in pixels.",
    )
    parser.add_argument(
        "--compression",
        default="lzw",
        choices=["lzw", "deflate", "jpeg", "none"],
        help="TIFF tile compression.",
    )
    parser.add_argument(
        "--fine-registration",
        action="store_true",
        default=False,
        help=(
            "Run the optional nuclei-level fine registration step.  "
            "Requires 'fixed_nuclei_csv' and 'moving_nuclei_csv' columns in "
            "the CSV (or the pair will skip fine registration with a warning)."
        ),
    )
    parser.add_argument(
        "--no-visualise",
        dest="visualise",
        action="store_false",
        default=True,
        help="Suppress saving the checkerboard overlay PNG for each pair.",
    )

    return parser


def main(argv: Optional[list] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    run_batch(
        csv_path=args.csv,
        output_dir=args.output_dir,
        source_magnification=args.source_mag,
        target_magnification=args.target_mag,
        tile_size=args.tile_size,
        compression=args.compression,
        run_fine_registration=args.fine_registration,
        visualise=args.visualise,
    )


if __name__ == "__main__":
    main()
