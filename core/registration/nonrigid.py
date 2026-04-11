import functools
import logging
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from typing import Optional, Tuple, List, Union, Dict, Any
import numpy as np
import math
import cv2
from skimage import color
from pathlib import Path
from tqdm.auto import tqdm
from core.config import PREPROCESSING_RESOLUTION, REGISTRATION_RESOLUTION

logger = logging.getLogger(__name__)

# Scale factor relating coarse (preprocessing) resolution to fine (registration) resolution.
# Used to convert nuclei coordinates between the two resolution spaces.
_RESOLUTION_SCALE: float = PREPROCESSING_RESOLUTION / REGISTRATION_RESOLUTION


@functools.lru_cache(maxsize=32)
def _make_gaussian_blur(kernel_width: int, blur_sigma: float) -> transforms.GaussianBlur:
    """Return a cached GaussianBlur transform for the given parameters."""
    return transforms.GaussianBlur(kernel_width, blur_sigma)


def gaussian_smoothing(input_tensor: torch.Tensor, blur_sigma: float) -> torch.Tensor:
    """Apply Gaussian blur to a tensor. GaussianBlur objects are cached to avoid re-computing kernels."""
    with torch.set_grad_enabled(False):
        kernel_width = int(blur_sigma * 2.54) + 1
        if kernel_width % 2 == 0:
            kernel_width += 1
        return _make_gaussian_blur(kernel_width, blur_sigma)(input_tensor)


def initialize_deformation_field(input_tensor: torch.Tensor) -> torch.Tensor:
    """Initialize a zero deformation field matching the spatial dimensions of the input tensor."""
    dim_count = len(input_tensor.size()) - 2
    return torch.zeros(
        (input_tensor.size(0), input_tensor.size(2), input_tensor.size(3)) + (dim_count,)
    ).type_as(input_tensor)


def build_reference_coordinate_system(input_tensor: Optional[torch.Tensor] = None,
                                       dimensions: Optional[torch.Size] = None,
                                       compute_device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    """Build a reference coordinate grid for the given tensor dimensions."""
    if input_tensor is not None:
        dimensions = input_tensor.size()

    # Convert string device specification to torch.device
    if isinstance(compute_device, str):
        compute_device = torch.device(compute_device)

    if compute_device is None and input_tensor is not None:
        base_transform = torch.eye(len(dimensions)-1)[:-1, :].unsqueeze(0).type_as(input_tensor)
    else:
        base_transform = torch.eye(len(dimensions)-1, device=compute_device)[:-1, :].unsqueeze(0)

    base_transform = torch.repeat_interleave(base_transform, dimensions[0], dim=0)
    coordinate_grid = F.affine_grid(base_transform, dimensions, align_corners=False)

    return coordinate_grid


def compute_smoothness_regularization(vector_field: "torch.Tensor",
                               compute_device: "torch.device" = None,
                               weight_map: "Optional[torch.Tensor]" = None) -> "torch.Tensor":
    dim_count = len(vector_field.size()) - 2
    
    if dim_count == 2:
        x_grad = ((vector_field[:, 1:, :, :] - vector_field[:, :-1, :, :]) * 
              vector_field.shape[1])**2
        y_grad = ((vector_field[:, :, 1:, :] - vector_field[:, :, :-1, :]) * 
              vector_field.shape[2])**2
        
        if weight_map is not None:
            # Apply spatial weighting if provided
            x_weight = weight_map[:, 1:, :].unsqueeze(-1)
            y_weight = weight_map[:, :, 1:].unsqueeze(-1)
            smoothness_term = (torch.mean(x_grad * x_weight) + torch.mean(y_grad * y_weight)) / 2
        else:
            smoothness_term = (torch.mean(x_grad) + torch.mean(y_grad)) / 2
    else:
        raise ValueError("Unsupported dimensionality. Must be 2D or 3D.")
        
    return smoothness_term


def scale_tensor_to_dimensions(input_tensor: torch.Tensor,
                               target_dimensions: torch.Size,
                               interpolation_method: str = 'bilinear') -> torch.Tensor:
    """Resize input tensor to the given spatial dimensions."""
    return F.interpolate(input_tensor, size=target_dimensions,
                         mode=interpolation_method, align_corners=False)


def compute_normalized_cross_correlation(sources: torch.Tensor,
                                         targets: torch.Tensor,
                                         device: Optional[Union[str, torch.device]] = None,
                                         **config_params) -> torch.Tensor:
    ndim = len(sources.size()) - 2
    if ndim not in [2, 3]:
        raise ValueError("Unsupported number of dimensions.")
    size = config_params.get('size', 7)

    window = (size, ) * ndim
    if device is None:
        sum_filt = torch.ones([1, 1, *window]).type_as(sources)
    else:
        sum_filt = torch.ones([1, 1, *window], device=device)

    pad_no = math.floor(window[0] / 2)
    stride = ndim * (1,)
    padding = ndim * (pad_no,)
    conv_fn = getattr(F, 'conv%dd' % ndim)
    sources_denom = sources**2
    targets_denom = targets**2
    numerator = sources*targets
    sources_sum = conv_fn(sources, sum_filt, stride=stride, padding=padding)
    targets_sum = conv_fn(targets, sum_filt, stride=stride, padding=padding)
    sources_denom_sum = conv_fn(sources_denom, sum_filt, stride=stride, padding=padding)
    targets_denom_sum = conv_fn(targets_denom, sum_filt, stride=stride, padding=padding)
    numerator_sum = conv_fn(numerator, sum_filt, stride=stride, padding=padding)
    size = np.prod(window)
    u_sources = sources_sum / size
    u_targets = targets_sum / size
    cross = numerator_sum - u_targets * sources_sum - u_sources * targets_sum + u_sources * u_targets * size
    sources_var = sources_denom_sum - 2 * u_sources * sources_sum + u_sources * u_sources * size
    targets_var = targets_denom_sum - 2 * u_targets * targets_sum + u_targets * u_targets * size
    ncc = cross * cross / (sources_var * targets_var + 1e-5)
    return -torch.mean(ncc)


def apply_deformation_field(input_tensor: torch.Tensor,
                             vector_field: torch.Tensor,
                             coord_grid: Optional[torch.Tensor] = None,
                             interpolation_method: str = 'bilinear',
                             boundary_handling: str = 'zeros',
                             compute_device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    """Apply a deformation field to warp the input tensor."""
    # Convert string device specification to torch.device
    if isinstance(compute_device, str):
        compute_device = torch.device(compute_device)
        
    if coord_grid is None:
        coord_grid = build_reference_coordinate_system(input_tensor=input_tensor, compute_device=compute_device)
        
    sampling_coordinates = coord_grid + vector_field
    deformed_tensor = F.grid_sample(input_tensor, sampling_coordinates, 
                                    mode=interpolation_method, 
                                    padding_mode=boundary_handling, 
                                    align_corners=False)
    
    return deformed_tensor


def scale_deformation_field(vector_field: torch.Tensor,
                          new_dimensions: Union[torch.Size, Tuple[int, int]], 
                          interpolation_method: str = 'bilinear') -> torch.Tensor:

    # Permute to channel-first format for interpolation
    channel_first = vector_field.permute(0, 3, 1, 2)
    
    # Perform interpolation
    resized = F.interpolate(
        channel_first, 
        size=new_dimensions, 
        mode=interpolation_method, 
        align_corners=False
    )
    
    # Return to original format
    return resized.permute(0, 2, 3, 1)


def create_multiscale_representation(input_tensor: torch.Tensor,
                                      level_count: int,
                                      interpolation_method: str = 'bilinear',
                                      scale_factor: float = 2.0) -> List[torch.Tensor]:
    """Build a multi-scale image pyramid from fine to coarse."""
    pyramid_levels = [None] * level_count
    
    # Build from fine to coarse
    for i in range(level_count - 1, -1, -1):
        if i == level_count - 1:
            # Original resolution
            pyramid_levels[i] = input_tensor
        else:
            # Get previous level and compute dimensions for current level
            prev_size = pyramid_levels[i+1].size()
            current_dims = tuple(int(prev_size[j] / scale_factor) if j > 1 else prev_size[j] 
                               for j in range(len(prev_size)))
            
            # Extract just the spatial dimensions
            spatial_dims = torch.Size(current_dims)[2:]
            
            # Apply smoothing to prevent aliasing, then downsample
            smoothed = gaussian_smoothing(pyramid_levels[i+1], 1)
            downsampled = scale_tensor_to_dimensions(smoothed, spatial_dims, 
                                                 interpolation_method)
            
            pyramid_levels[i] = downsampled
            
    return pyramid_levels


def convert_image_to_tensor(img_array: np.ndarray, compute_device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    """Convert a numpy image array to a PyTorch tensor."""
    # Convert string device specification to torch.device
    if isinstance(compute_device, str):
        compute_device = torch.device(compute_device)
        
    # Normalize image if it's not already in [0, 1] range
    if img_array.dtype != np.float32 and img_array.dtype != np.float64:
        if img_array.max() > 1.0:
            img_array = img_array.astype(np.float32) / 255.0
    
    if len(img_array.shape) == 3:
        # Color image
        return torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(compute_device)
    elif len(img_array.shape) == 2:
        # Grayscale image
        return torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(compute_device)
    else:
        raise ValueError(f"Unsupported image dimensions: {img_array.shape}")


def prepare_image_tensors(source_image: np.ndarray, 
                        target_image: np.ndarray, 
                        compute_device: Union[str, torch.device],
                        normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:

    # Convert string device specification to torch.device
    if isinstance(compute_device, str):
        compute_device = torch.device(compute_device)
        
    # Convert to grayscale if RGB
    if len(source_image.shape) == 3 and source_image.shape[2] == 3:
        gray_source = color.rgb2gray(source_image)
    else:
        gray_source = source_image
        
    if len(target_image.shape) == 3 and target_image.shape[2] == 3:
        gray_target = color.rgb2gray(target_image)
    else:
        gray_target = target_image

    # Normalize if requested
    if normalize:
        gray_source = (gray_source - gray_source.min()) / (gray_source.max() - gray_source.min() + 1e-10)
        gray_target = (gray_target - gray_target.min()) / (gray_target.max() - gray_target.min() + 1e-10)

    # Convert to tensor format
    tensor_source = convert_image_to_tensor(gray_source, compute_device)
    tensor_target = convert_image_to_tensor(gray_target, compute_device)

    # Create tensors with gradient tracking
    source_tensor = tensor_source.clone().detach().to(dtype=torch.float32, device=compute_device).requires_grad_(True)
    target_tensor = tensor_target.clone().detach().to(dtype=torch.float32, device=compute_device).requires_grad_(True)

    return source_tensor, target_tensor

def elastic_image_registration(
    source: np.ndarray, 
    target: np.ndarray, 
    similarity_metric: str = "ncc",
    similarity_metric_params: Dict[str, Any] = {"size": 7},
    compute_device: Union[str, torch.device] = "cuda",
    verbose: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    save_intermediate: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Setup
    device = torch.device(compute_device) if isinstance(compute_device, str) else compute_device
    # Resize source to match target dimensions (preserving reflect border) before building pyramids
    aligned_source = cv2.warpAffine(source, np.eye(2, 3), (target.shape[1], target.shape[0]), borderMode=cv2.BORDER_REFLECT)
    source_t, target_t = prepare_image_tensors(aligned_source, target, device)

    pyramid_levels = 6
    src_pyr = create_multiscale_representation(source_t, pyramid_levels)
    tgt_pyr = create_multiscale_representation(target_t, pyramid_levels)

    # Hyperparameters
    iterations_per_level = [200, 200, 150, 100, 100, 80]
    learning_rates = [0.01, 0.005, 0.002, 0.002, 0.001, 0.0005]
    regularization_weights = [10.0, 1.5, 1.2, 1.0, 0.8, 0.4]
    smoothing_sigma = 1.25
    prev_def_field = None
    # Loop through pyramid levels
    for lvl in range(pyramid_levels):
        curr_src = src_pyr[lvl]
        curr_tgt = tgt_pyr[lvl]
        H, W = curr_src.shape[2:]

        # Initialize or upsample deformation field
        if lvl == 0:
            def_field = initialize_deformation_field(curr_src).detach().clone().requires_grad_(True)
        else:
            def_field = scale_deformation_field(prev_def_field, (H, W)).detach().clone().requires_grad_(True)

        # Optimizer: LBFGS on final level, Adam otherwise
        if lvl == pyramid_levels - 1:
            optimizer = optim.LBFGS([def_field], lr=learning_rates[lvl], max_iter=50, line_search_fn="strong_wolfe")
        else:
            optimizer = optim.Adam([def_field], lr=learning_rates[lvl])

        weight = regularization_weights[lvl]

        for iter_idx in tqdm(range(iterations_per_level[lvl]), disable=not verbose, desc=f"Level {lvl}/{pyramid_levels-1}"):
            def closure():
                optimizer.zero_grad()
                warped = apply_deformation_field(curr_src, def_field, compute_device=device)
                sim_loss = compute_normalized_cross_correlation(warped, curr_tgt, compute_device=device, **similarity_metric_params)
                reg_loss = deformation_loss(def_field, compute_device=device)
                loss = sim_loss + weight * reg_loss
                loss.backward()
                return loss

            loss = optimizer.step(closure)

            with torch.no_grad():
                # Optional clipping to prevent folding
                max_disp = 5.0  # pixels
                def_field.clamp_(-max_disp, max_disp)

        prev_def_field = def_field

    # Upsample to original shape if needed
    final_def = scale_deformation_field(prev_def_field, (source_t.size(2), source_t.size(3))) \
        if tuple(prev_def_field.shape[1:3]) != (source_t.size(2), source_t.size(3)) else prev_def_field
    final_warped = apply_deformation_field(source_t, final_def, compute_device=device)

    if save_intermediate and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        warped_np = (final_warped.detach().cpu().numpy()[0, 0] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, "final_warped.png"), warped_np)
        logger.info("Saved intermediate warped image to %s", output_dir)

    return final_def, final_warped


def compute_deformation_and_apply(
    source_prep,
    final_transform,
    displacement_field,
    moving_df,
    fixed_df,
    padding_params,
    util,
    pad_landmarks,
):
    """
    Compute the final deformation field by combining rigid and non-rigid transformations,
    then apply it to the moving landmark points.

    Parameters
    ----------
    source_prep : np.ndarray
        Preprocessed source image used for rigid transformation.
    final_transform : object
        Transformation model or matrix from rigid registration.
    displacement_field : torch.Tensor or np.ndarray
        The predicted displacement field (2D vector field).
    moving_df : pandas.DataFrame
        DataFrame containing moving landmark coordinates with columns ['global_x', 'global_y'].
    fixed_df : pandas.DataFrame
        DataFrame containing fixed landmark coordinates with columns ['global_x', 'global_y'].
    padding_params : tuple
        Padding parameters required by `pad_landmarks`.
    util : module
        Utility module containing helper functions:
        - rigid_dot
        - tc_df_to_np_df
        - compose_vector_fields
        - apply_deformation_to_points
    pad_landmarks : callable
        Function to pad landmark coordinates to match deformation field dimensions.

    Returns
    -------
    deformation_field : np.ndarray
        Combined deformation field, shape (2, H, W).
    moving_updated : np.ndarray
        Updated (deformed) moving points, scaled back to original coordinates.
    fixed_points : np.ndarray
        Fixed points, scaled back to original coordinates.
    moving_points : np.ndarray
        Original moving points, scaled back to original coordinates.
    """

    # Step 1: Rigid transformation
    i_x, i_y = util.rigid_dot(source_prep, final_transform)

    # Step 2: Convert and compose with displacement field
    disp_field_np = util.tc_df_to_np_df(displacement_field)
    r_x, r_y = util.compose_vector_fields(i_x, i_y, disp_field_np[0], disp_field_np[1])
    deformation_field = np.stack((r_x, r_y), axis=0)

    logger.info("Deformation field shape: %s", deformation_field.shape)

    # Step 3: Prepare landmark coordinates
    # Scale from full-resolution pixel space down to the coarse registration resolution.
    moving_points = moving_df[['global_x', 'global_y']].values * _RESOLUTION_SCALE
    fixed_points = fixed_df[['global_x', 'global_y']].values * _RESOLUTION_SCALE
    moving_points, fixed_points = pad_landmarks(padding_params, moving_points, fixed_points)

    # Step 4: Apply deformation
    moving_updated = util.apply_deformation_to_points(moving_points, deformation_field)

    # Step 5: Scale back to original pixel space
    fixed_points = fixed_points / _RESOLUTION_SCALE
    moving_points = moving_points / _RESOLUTION_SCALE
    moving_updated = moving_updated / _RESOLUTION_SCALE

    return deformation_field, moving_updated, fixed_points, moving_points
