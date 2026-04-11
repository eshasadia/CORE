"""
MHA Deformation Field Application for CORE

Apply deformation fields computed at low resolution (e.g., 0.625x) 
to whole slide images at high resolution (e.g., 40x).

Features:
- Uses pyvips for memory-efficient large image handling
- Scales deformation fields from low to high resolution
- Properly scales both spatial dimensions and displacement magnitudes
"""

import numpy as np
import pyvips
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, Optional, Union
import argparse


def get_image_shape(img: Union[np.ndarray, pyvips.Image]) -> Tuple[int, ...]:
    """Get shape of image (row, col) or (row, col, channels)."""
    if isinstance(img, pyvips.Image):
        if img.bands > 1:
            return (img.height, img.width, img.bands)
        return (img.height, img.width)
    return img.shape


def array_to_vips(arr: np.ndarray) -> pyvips.Image:
    """Convert numpy array to pyvips Image."""
    if arr.ndim == 2:
        height, width = arr.shape
        bands = 1
        linear = arr.flatten()
    else:
        height, width, bands = arr.shape
        linear = arr.flatten()
    
    dtype_to_format = {
        np.dtype('uint8'): 'uchar',
        np.dtype('int8'): 'char',
        np.dtype('uint16'): 'ushort',
        np.dtype('int16'): 'short',
        np.dtype('uint32'): 'uint',
        np.dtype('int32'): 'int',
        np.dtype('float32'): 'float',
        np.dtype('float64'): 'double',
    }
    
    vips_format = dtype_to_format.get(arr.dtype, 'float')
    
    vips_img = pyvips.Image.new_from_memory(
        linear.tobytes(),
        width, height, bands,
        vips_format
    )
    
    return vips_img


def vips_to_array(vips_img: pyvips.Image) -> np.ndarray:
    """Convert pyvips Image to numpy array."""
    format_map = {
        'uchar': np.uint8,
        'char': np.int8,
        'ushort': np.uint16,
        'short': np.int16,
        'uint': np.uint32,
        'int': np.int32,
        'float': np.float32,
        'double': np.float64,
    }
    
    dtype = np.dtype(format_map.get(vips_img.format, np.float32))
    
    np_arr = np.ndarray(
        buffer=vips_img.write_to_memory(),
        dtype=dtype,
        shape=[vips_img.height, vips_img.width, vips_img.bands]
    )
    
    if np_arr.shape[2] == 1:
        np_arr = np_arr.squeeze(axis=2)
    
    return np_arr


def resize_displacement_field(
    displacement: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    target_shape_rc: Tuple[int, int],
    source_shape_rc: Tuple[int, int],
    registered_shape_rc: Tuple[int, int]
) -> np.ndarray:
    """
    Resize displacement field to match target resolution.
    
    Steps:
    1. Resize the displacement field spatially to match target dimensions
    2. Scale the displacement values by the ratio of resolutions
    
    Args:
        displacement: Displacement field as (2, H, W) array or tuple of (dx, dy)
        target_shape_rc: Target output shape (rows, cols)
        source_shape_rc: Shape of image where transform was computed
        registered_shape_rc: Shape of registered image at transform resolution
    
    Returns:
        Resized displacement field as (2, H, W) array
    """
    # Convert tuple to array if needed
    if isinstance(displacement, tuple):
        displacement = np.array([displacement[0], displacement[1]])
    
    # Calculate scaling factors
    scale_factor_rc = np.array(target_shape_rc) / np.array(registered_shape_rc)
    
    # Extract x and y components
    disp_x = displacement[0]
    disp_y = displacement[1]
    
    # Use pyvips for efficient resizing
    vips_disp_x = array_to_vips(disp_x.astype(np.float32))
    vips_disp_y = array_to_vips(disp_y.astype(np.float32))
    
    # Resize to target dimensions
    scale_x = target_shape_rc[1] / disp_x.shape[1]
    scale_y = target_shape_rc[0] / disp_x.shape[0]
    
    resized_disp_x = vips_disp_x.resize(scale_x, vscale=scale_y)
    resized_disp_y = vips_disp_y.resize(scale_x, vscale=scale_y)
    
    # Convert back to numpy
    scaled_disp_x = vips_to_array(resized_disp_x)
    scaled_disp_y = vips_to_array(resized_disp_y)
    
    # Scale the displacement values
    scaled_disp_x = scaled_disp_x * scale_factor_rc[1]  # x by column scale
    scaled_disp_y = scaled_disp_y * scale_factor_rc[0]  # y by row scale
    
    return np.array([scaled_disp_x, scaled_disp_y])


def apply_displacement_field(
    img: Union[np.ndarray, pyvips.Image],
    displacement: np.ndarray,
    output_shape_rc: Tuple[int, int],
    source_shape_rc: Tuple[int, int],
    registered_shape_rc: Tuple[int, int],
    background: Optional[Union[int, float, list]] = None,
    interpolation: str = "bicubic"
) -> pyvips.Image:
    """
    Apply displacement field to warp an image.
    
    Args:
        img: Input image to warp
        displacement: Backwards displacement field (2, H, W)
        output_shape_rc: Output shape (rows, cols)
        source_shape_rc: Shape where transform was computed
        registered_shape_rc: Shape of registered image at transform resolution
        background: Background color for areas outside the image
        interpolation: Interpolation method
    
    Returns:
        Warped pyvips Image
    """
    # Convert numpy to vips if needed
    if isinstance(img, np.ndarray):
        vips_img = array_to_vips(img)
    else:
        vips_img = img
    
    # Get image properties
    n_channels = vips_img.bands
    
    # Set default background
    if background is None:
        background = [0] * n_channels
    elif isinstance(background, (int, float)):
        background = [background] * n_channels
    
    # Resize displacement field to output dimensions
    scaled_displacement = resize_displacement_field(
        displacement,
        output_shape_rc,
        source_shape_rc,
        registered_shape_rc
    )
    
    # Create coordinate grids for mapping
    rows, cols = output_shape_rc
    
    col_indices = np.tile(np.arange(cols, dtype=np.float32), (rows, 1))
    row_indices = np.tile(np.arange(rows, dtype=np.float32).reshape(-1, 1), (1, cols))
    
    # Add displacements to get source coordinates
    source_cols = col_indices + scaled_displacement[0]
    source_rows = row_indices + scaled_displacement[1]
    
    # Stack into index image for mapim (2 bands: x, y)
    coord_map = np.stack([source_cols, source_rows], axis=2).astype(np.float32)
    vips_coord_map = array_to_vips(coord_map)
    
    # Set interpolation method
    interp_methods = {
        'nearest': pyvips.Interpolate.new('nearest'),
        'bilinear': pyvips.Interpolate.new('bilinear'),
        'bicubic': pyvips.Interpolate.new('bicubic'),
    }
    interpolator = interp_methods.get(interpolation, pyvips.Interpolate.new('bicubic'))
    
    # Warp image using coordinate mapping
    warped = vips_img.mapim(vips_coord_map, interpolate=interpolator, background=background)
    
    return warped


def load_mha_displacement(mha_path: str) -> Tuple[np.ndarray, dict]:
    """
    Load MHA deformation field and convert to displacement array format.
    
    Output format is (2, H, W) where:
    - displacement[0] = dx (column/x displacement)
    - displacement[1] = dy (row/y displacement)
    
    Args:
        mha_path: Path to MHA file
    
    Returns:
        Tuple of (displacement array, metadata dict)
    """
    print(f"Loading deformation field: {mha_path}")
    
    sitk_img = sitk.ReadImage(mha_path)
    
    # Get metadata
    metadata = {
        'size': sitk_img.GetSize(),
        'spacing': sitk_img.GetSpacing(),
        'origin': sitk_img.GetOrigin(),
        'direction': sitk_img.GetDirection(),
    }
    
    print(f"  Size: {metadata['size']}")
    print(f"  Spacing: {metadata['spacing']}")
    
    # Convert to numpy
    arr = sitk.GetArrayFromImage(sitk_img)
    
    # Handle different array shapes
    if arr.ndim == 3:
        if arr.shape[2] == 2:
            # (H, W, 2) -> (2, H, W)
            disp_x = arr[:, :, 0]
            disp_y = arr[:, :, 1]
        elif arr.shape[0] == 2:
            # Already (2, H, W)
            disp_x = arr[0]
            disp_y = arr[1]
        else:
            raise ValueError(f"Unexpected deformation field shape: {arr.shape}")
    else:
        raise ValueError(f"Expected 3D array, got shape: {arr.shape}")
    
    displacement = np.array([disp_x, disp_y], dtype=np.float32)
    print(f"  Displacement shape: {displacement.shape}")
    
    return displacement, metadata


def register_wsi_with_mha(
    mha_path: str,
    wsi_path: str,
    output_path: str,
    source_magnification: float = 0.625,
    target_magnification: float = 40.0,
    interpolation: str = "bicubic",
    background: int = 0
):
    """
    Apply MHA deformation field to WSI for registration.
    
    Args:
        mha_path: Path to deformation field MHA file
        wsi_path: Path to WSI to be warped
        output_path: Output path for registered image
        source_magnification: Magnification at which deformation was computed
        target_magnification: Target WSI magnification
        interpolation: Interpolation method
        background: Background color
    """
    print("=" * 60)
    print("WSI Registration with MHA Deformation Field")
    print("=" * 60)
    
    # Calculate scale factor
    scale_factor = target_magnification / source_magnification
    print(f"Scale factor: {scale_factor}x ({source_magnification}x -> {target_magnification}x)")
    print()
    
    # Load deformation field
    displacement, mha_metadata = load_mha_displacement(mha_path)
    source_shape_rc = (displacement.shape[1], displacement.shape[2])
    registered_shape_rc = source_shape_rc
    print()
    
    # Load WSI with pyvips for memory efficiency
    print(f"Loading WSI: {wsi_path}")
    vips_img = pyvips.Image.new_from_file(wsi_path, access='sequential')
    print(f"  Size: {vips_img.width} x {vips_img.height}")
    print(f"  Bands: {vips_img.bands}")
    print()
    
    # Calculate output shape
    output_shape_rc = (vips_img.height, vips_img.width)
    
    # Apply deformation
    print("Applying deformation field...")
    warped = apply_displacement_field(
        img=vips_img,
        displacement=displacement,
        output_shape_rc=output_shape_rc,
        source_shape_rc=source_shape_rc,
        registered_shape_rc=registered_shape_rc,
        background=background,
        interpolation=interpolation
    )
    print("  Warping complete")
    print()
    
    # Save result
    print(f"Saving to: {output_path}")
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Determine output format based on extension
    ext = Path(output_path).suffix.lower()
    
    if ext == '.mha':
        # Convert to numpy and save with SimpleITK
        warped_arr = vips_to_array(warped)
        sitk_img = sitk.GetImageFromArray(warped_arr)
        sitk.WriteImage(sitk_img, output_path, useCompression=True)
    elif output_path.endswith('.ome.tiff') or output_path.endswith('.ome.tif'):
        # Save as OME-TIFF (pyramidal) — must be checked before plain .tiff
        warped.write_to_file(
            output_path,
            tile=True,
            pyramid=True,
            compression='lzw',
            tile_width=256,
            tile_height=256
        )
    elif ext in ['.tiff', '.tif']:
        # Save as TIFF with pyvips
        warped.write_to_file(output_path, compression='lzw')
    else:
        # Default: save with pyvips
        warped.write_to_file(output_path)
    
    print("  Save complete")
    print()
    print("=" * 60)
    print("Registration complete!")
    print("=" * 60)


def save_displacement_as_mha(
    displacement: np.ndarray,
    output_path: str,
    spacing: Tuple[float, float] = (1.0, 1.0),
    origin: Tuple[float, float] = (0.0, 0.0)
):
    """
    Save displacement array as MHA file.
    
    Args:
        displacement: Displacement field (2, H, W)
        output_path: Output path
        spacing: Pixel spacing
        origin: Image origin
    """
    # Convert (2, H, W) to (H, W, 2) for SimpleITK
    disp_x = displacement[0]
    disp_y = displacement[1]
    arr = np.stack([disp_x, disp_y], axis=2).astype(np.float64)
    
    # Create SimpleITK image
    sitk_img = sitk.GetImageFromArray(arr, isVector=True)
    sitk_img.SetSpacing(spacing)
    sitk_img.SetOrigin(origin)
    
    # Save
    sitk.WriteImage(sitk_img, output_path, useCompression=True)
    print(f"Saved displacement field to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply MHA deformation field to WSI for registration"
    )
    parser.add_argument(
        "--mha", "-m",
        required=True,
        help="Path to MHA deformation field"
    )
    parser.add_argument(
        "--wsi", "-w",
        required=True,
        help="Path to WSI to be registered"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output path"
    )
    parser.add_argument(
        "--source-mag",
        type=float,
        default=0.625,
        help="Source magnification (default: 0.625)"
    )
    parser.add_argument(
        "--target-mag",
        type=float,
        default=40.0,
        help="Target magnification (default: 40.0)"
    )
    parser.add_argument(
        "--interp",
        default="bicubic",
        choices=["nearest", "bilinear", "bicubic"],
        help="Interpolation method"
    )
    
    args = parser.parse_args()
    
    register_wsi_with_mha(
        mha_path=args.mha,
        wsi_path=args.wsi,
        output_path=args.output,
        source_magnification=args.source_mag,
        target_magnification=args.target_mag,
        interpolation=args.interp
    )
