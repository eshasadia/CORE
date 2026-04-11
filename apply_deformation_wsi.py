"""
apply_deformation_wsi.py
========================
Apply a pre-computed MHA deformation field to a whole-slide image (WSI)
at full resolution and save the result as a pyramidal OME-TIFF.

The deformation field is typically computed at a low magnification
(e.g., 0.625x).  This script scales the field both spatially and in
displacement magnitude to match the full-resolution WSI, then applies
the warp using pyvips so that the image is never fully loaded into RAM.

Usage
-----
    python apply_deformation_wsi.py \\
        --mha   deformation_field.mha \\
        --wsi   source_slide.tiff \\
        --output registered_slide.ome.tiff \\
        [--source-mag  0.625] \\
        [--target-mag  40.0] \\
        [--tile-size   512] \\
        [--compression lzw] \\
        [--interp      bicubic] \\
        [--background  0]

Dependencies
------------
    pyvips   – memory-efficient image I/O and warping
    SimpleITK – reading MHA deformation fields
    numpy    – displacement field manipulation

Install via::

    pip install pyvips SimpleITK numpy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pyvips
import SimpleITK as sitk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_mha_displacement(mha_path: str) -> Tuple[np.ndarray, dict]:
    """
    Load an MHA deformation field and return a (2, H, W) float32 array.

    The two channels are:
      * ``displacement[0]``  — dx  (column / x displacement)
      * ``displacement[1]``  — dy  (row / y displacement)

    Parameters
    ----------
    mha_path:
        Path to the ``.mha`` file produced by the CORE registration pipeline.

    Returns
    -------
    displacement:
        Shape (2, H, W) float32.
    metadata:
        Dict with ``size``, ``spacing``, ``origin``, and ``direction``.
    """
    print(f"[1/4] Loading deformation field: {mha_path}")
    sitk_img = sitk.ReadImage(mha_path)

    metadata = {
        "size":      sitk_img.GetSize(),
        "spacing":   sitk_img.GetSpacing(),
        "origin":    sitk_img.GetOrigin(),
        "direction": sitk_img.GetDirection(),
    }
    print(f"      MHA size    : {metadata['size']}")
    print(f"      MHA spacing : {metadata['spacing']}")

    arr = sitk.GetArrayFromImage(sitk_img)  # (H, W, 2) or (2, H, W)

    if arr.ndim == 3 and arr.shape[2] == 2:
        # (H, W, 2) → (2, H, W)
        disp_x = arr[:, :, 0]
        disp_y = arr[:, :, 1]
    elif arr.ndim == 3 and arr.shape[0] == 2:
        # Already (2, H, W)
        disp_x = arr[0]
        disp_y = arr[1]
    else:
        raise ValueError(
            f"Unexpected deformation field shape: {arr.shape}. "
            "Expected (H, W, 2) or (2, H, W)."
        )

    displacement = np.array([disp_x, disp_y], dtype=np.float32)
    print(f"      Displacement shape: {displacement.shape}")
    return displacement, metadata


def _array_to_vips(arr: np.ndarray) -> pyvips.Image:
    """Convert a 2-D float32 numpy array to a single-band pyvips Image."""
    assert arr.ndim == 2, "Expected 2-D array"
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return pyvips.Image.new_from_memory(
        arr.tobytes(), arr.shape[1], arr.shape[0], 1, "float"
    )


def _build_coord_map_vips(
    displacement: np.ndarray,
    wsi_width: int,
    wsi_height: int,
    disp_width: int,
    disp_height: int,
) -> pyvips.Image:
    """
    Build a pyvips 2-band coordinate map for ``mapim``.

    The coordinate map encodes, for every output pixel ``(col, row)``,
    the source pixel coordinates ``(src_col, src_row)`` in the WSI:

        src_col = col + scaled_dx(col, row)
        src_row = row + scaled_dy(col, row)

    Both the spatial resize of the displacement field and the scaling of
    displacement *values* are done entirely inside pyvips so that
    multi-gigapixel WSIs are processed without ever materialising a
    full-resolution numpy array.

    Parameters
    ----------
    displacement:
        Low-resolution displacement field, shape (2, H_disp, W_disp).
    wsi_width, wsi_height:
        Target dimensions of the full-resolution WSI.
    disp_width, disp_height:
        Spatial dimensions of the low-res displacement field.

    Returns
    -------
    pyvips.Image
        2-band float image of shape (wsi_height, wsi_width).
    """
    # Scale factors: how much larger the full-res WSI is vs. the disp. field
    scale_x = wsi_width  / disp_width
    scale_y = wsi_height / disp_height

    # --- Scale displacement field spatially (resize) and in value ----------
    # displacement[0] = dx  (columns)
    # displacement[1] = dy  (rows)
    disp_x_vips = _array_to_vips(displacement[0])
    disp_y_vips = _array_to_vips(displacement[1])

    # Resize to full WSI dimensions
    disp_x_full = disp_x_vips.resize(scale_x, vscale=scale_y)
    disp_y_full = disp_y_vips.resize(scale_x, vscale=scale_y)

    # Scale the *values* to match the new pixel pitch
    disp_x_full = disp_x_full * scale_x
    disp_y_full = disp_y_full * scale_y

    # --- Build base coordinate grids with pyvips.Image.xyz ----------------
    # xyz() returns a 2-band image: band-0 = x (col), band-1 = y (row)
    xyz = pyvips.Image.xyz(wsi_width, wsi_height)
    base_x = xyz.extract_band(0).cast("float")
    base_y = xyz.extract_band(1).cast("float")

    # Source coordinates
    src_x = base_x + disp_x_full
    src_y = base_y + disp_y_full

    # Stack into 2-band coordinate map expected by mapim
    coord_map = src_x.bandjoin(src_y)
    return coord_map


def _save_ome_tiff(
    warped: pyvips.Image,
    output_path: str,
    tile_size: int,
    compression: str,
) -> None:
    """
    Save a pyvips Image as a pyramidal OME-TIFF.

    The file is written as a tiled, multi-resolution (sub-IFD pyramid)
    BigTIFF that is compatible with OME-TIFF viewers.

    Parameters
    ----------
    warped:
        The registered pyvips image.
    output_path:
        Destination file path (should end with ``.ome.tiff`` or ``.ome.tif``).
    tile_size:
        Tile size in pixels (used for both width and height).
    compression:
        pyvips compression string, e.g. ``'lzw'``, ``'deflate'``, ``'jpeg'``,
        or ``'none'``.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Try to write with tifffile for proper OME-XML metadata if available
    try:
        import tifffile  # noqa: F401
        _save_ome_tiff_via_tifffile(warped, output_path, tile_size, compression)
    except ImportError:
        _save_ome_tiff_via_pyvips(warped, output_path, tile_size, compression)


def _save_ome_tiff_via_pyvips(
    warped: pyvips.Image,
    output_path: str,
    tile_size: int,
    compression: str,
) -> None:
    """Write a pyramidal tiled TIFF using pyvips (OME-compatible)."""
    print(f"[4/4] Saving pyramidal TIFF (pyvips): {output_path}")
    warped.write_to_file(
        output_path,
        bigtiff=True,
        tile=True,
        pyramid=True,
        compression=compression,
        tile_width=tile_size,
        tile_height=tile_size,
        subifd=True,          # embed pyramid levels as sub-IFDs (OME convention)
    )
    print("      Save complete ✓")


def _save_ome_tiff_via_tifffile(
    warped: pyvips.Image,
    output_path: str,
    tile_size: int,
    compression: str,
) -> None:
    """
    Write a proper OME-TIFF with XML metadata using tifffile.

    Pyramid levels are computed by successive 2× downsampling via pyvips
    so that no level is materialised in RAM in full before writing.
    """
    import tifffile

    print(f"[4/4] Saving pyramidal OME-TIFF (tifffile): {output_path}")

    tifffile_compression = {
        "lzw": "lzw",
        "deflate": "deflate",
        "zlib": "deflate",
        "jpeg": "jpeg",
        "none": None,
    }.get(compression.lower(), "lzw")

    # Build pyramid levels as numpy arrays
    levels: list[np.ndarray] = []
    current = warped
    while True:
        # Convert current level to numpy
        fmt_map = {
            "uchar":  np.uint8,  "char":   np.int8,
            "ushort": np.uint16, "short":  np.int16,
            "uint":   np.uint32, "int":    np.int32,
            "float":  np.float32,"double": np.float64,
        }
        dtype = fmt_map.get(current.format, np.uint8)
        arr = np.ndarray(
            buffer=current.write_to_memory(),
            dtype=dtype,
            shape=(current.height, current.width, current.bands),
        )
        if current.bands == 1:
            arr = arr[:, :, 0]
        levels.append(arr)

        # Stop when the level is smaller than two tiles
        if current.width <= tile_size * 2 or current.height <= tile_size * 2:
            break
        current = current.resize(0.5)

    # Determine number of channels / axes
    base = levels[0]
    if base.ndim == 2:
        # Grayscale → (1, 1, H, W) for OME axes TCYX
        ome_data = [lvl[np.newaxis, np.newaxis] for lvl in levels]
    else:
        # RGB/multi-channel → (1, C, H, W)
        ome_data = [lvl.transpose(2, 0, 1)[np.newaxis] for lvl in levels]

    with tifffile.TiffWriter(output_path, bigtiff=True, ome=True) as tif:
        options = dict(
            tile=(tile_size, tile_size),
            compression=tifffile_compression,
            metadata=None,   # will be set by ome=True in TiffWriter
        )
        tif.write(ome_data[0], subifds=len(ome_data) - 1, **options)
        for sub_level in ome_data[1:]:
            tif.write(sub_level, subfiletype=1, **options)

    print("      Save complete ✓")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def apply_deformation_to_wsi(
    mha_path: str,
    wsi_path: str,
    output_path: str,
    source_magnification: float = 0.625,
    target_magnification: float = 40.0,
    tile_size: int = 512,
    compression: str = "lzw",
    interpolation: str = "bicubic",
    background: Union[int, float] = 0,
) -> None:
    """
    Apply an MHA deformation field to a WSI at full resolution and save
    the result as a pyramidal OME-TIFF.

    Parameters
    ----------
    mha_path:
        Path to the ``.mha`` deformation field produced by the CORE pipeline.
    wsi_path:
        Path to the moving (source) whole-slide image to be registered.
    output_path:
        Destination path for the registered image.  Should end with
        ``.ome.tiff`` or ``.ome.tif``.
    source_magnification:
        Magnification at which the deformation field was computed.
        Defaults to ``0.625``.
    target_magnification:
        Magnification of the WSI to be registered.  Defaults to ``40.0``.
    tile_size:
        Output tile size in pixels (for the pyramidal TIFF).
    compression:
        TIFF compression.  Choices: ``'lzw'`` (default), ``'deflate'``,
        ``'jpeg'``, ``'none'``.
    interpolation:
        Interpolation method for warping: ``'bicubic'`` (default),
        ``'bilinear'``, ``'nearest'``.
    background:
        Fill value for regions outside the original image boundary.
    """
    print("=" * 60)
    print("  WSI Deformation Field Application")
    print("=" * 60)
    print(f"  MHA path   : {mha_path}")
    print(f"  WSI path   : {wsi_path}")
    print(f"  Output     : {output_path}")
    print(
        f"  Scale      : {source_magnification}x → {target_magnification}x "
        f"({target_magnification / source_magnification:.1f}×)"
    )
    print()

    # ── 1. Load displacement field ──────────────────────────────────────────
    displacement, _ = _load_mha_displacement(mha_path)
    disp_height, disp_width = displacement.shape[1], displacement.shape[2]

    # ── 2. Load WSI ─────────────────────────────────────────────────────────
    print(f"[2/4] Loading WSI: {wsi_path}")
    # 'sequential' access lets pyvips stream the file without caching the
    # entire image in RAM.
    vips_img = pyvips.Image.new_from_file(wsi_path, access="sequential")
    wsi_width  = vips_img.width
    wsi_height = vips_img.height
    n_channels = vips_img.bands
    print(f"      Size    : {wsi_width} × {wsi_height}")
    print(f"      Channels: {n_channels}")
    print()

    # Background list must have one value per channel
    bg = [background] * n_channels if isinstance(background, (int, float)) else background

    # ── 3. Build coordinate map and warp ─────────────────────────────────────
    print("[3/4] Building coordinate map and applying deformation field ...")
    coord_map = _build_coord_map_vips(
        displacement,
        wsi_width,
        wsi_height,
        disp_width,
        disp_height,
    )

    interp_map = {
        "nearest":  pyvips.Interpolate.new("nearest"),
        "bilinear": pyvips.Interpolate.new("bilinear"),
        "bicubic":  pyvips.Interpolate.new("bicubic"),
    }
    interpolator = interp_map.get(interpolation, pyvips.Interpolate.new("bicubic"))

    # Re-open with random access for mapim (sequential is not compatible with
    # arbitrary coordinate lookups performed by mapim).
    vips_img_rand = pyvips.Image.new_from_file(wsi_path, access="random")

    warped = vips_img_rand.mapim(
        coord_map,
        interpolate=interpolator,
        background=bg,
    )
    print("      Warp complete ✓")
    print()

    # ── 4. Save as pyramidal OME-TIFF ─────────────────────────────────────────
    _save_ome_tiff(warped, output_path, tile_size=tile_size, compression=compression)

    print()
    print("=" * 60)
    print("  Registration complete!")
    print(f"  Output : {output_path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a pre-computed MHA deformation field to a WSI at full "
            "resolution and save the result as a pyramidal OME-TIFF."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mha", "-m",
        required=True,
        help="Path to the MHA deformation field.",
    )
    parser.add_argument(
        "--wsi", "-w",
        required=True,
        help="Path to the moving (source) WSI to be registered.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output path for the registered OME-TIFF (e.g., registered.ome.tiff).",
    )
    parser.add_argument(
        "--source-mag",
        type=float,
        default=0.625,
        help="Magnification at which the deformation field was computed.",
    )
    parser.add_argument(
        "--target-mag",
        type=float,
        default=40.0,
        help="Magnification of the input WSI.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Output tile size in pixels (for pyramidal TIFF).",
    )
    parser.add_argument(
        "--compression",
        default="lzw",
        choices=["lzw", "deflate", "jpeg", "none"],
        help="TIFF tile compression.",
    )
    parser.add_argument(
        "--interp",
        default="bicubic",
        choices=["nearest", "bilinear", "bicubic"],
        help="Interpolation method for pixel warping.",
    )
    parser.add_argument(
        "--background",
        type=float,
        default=0,
        help="Background fill value for out-of-bounds pixels.",
    )

    return parser


def main(argv: Optional[list] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    apply_deformation_to_wsi(
        mha_path=args.mha,
        wsi_path=args.wsi,
        output_path=args.output,
        source_magnification=args.source_mag,
        target_magnification=args.target_mag,
        tile_size=args.tile_size,
        compression=args.compression,
        interpolation=args.interp,
        background=args.background,
    )


if __name__ == "__main__":
    main()
