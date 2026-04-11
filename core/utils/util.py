import os
import cv2
import math
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
import csv
import torch
import numpy as np
from skimage import exposure, filters, img_as_float, color, measure, morphology
from skimage.registration import phase_cross_correlation
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy import ndimage as nd
from scipy.ndimage import map_coordinates
from tiatoolbox.utils.metrics import dice
from tiatoolbox import logger, rcParam
from scipy.interpolate import griddata

RGB_IMAGE_DIM = 3
BIN_MASK_DIM = 2

#  UTILS
def skip_subsample(points, n_samples=1000):
    total_points = points.shape[0]
    if total_points <= n_samples:
        return points
    step = total_points // n_samples
    return points[::step][:n_samples]

def gamma_corrections(img, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)
         
def tensor_to_rgb_numpy(tensor):
    # (1, 1, H, W) -> (3, H, W) -> (H, W, 3)
    tensor_rgb = tensor.squeeze().repeat(3, 1, 1)
    return tensor_rgb.permute(1, 2, 0).detach().cpu().numpy()
    
# DEFORMATION
def create_deformation_field(shape_transform, source_prep, u_x, u_y, util, output_path='./533_finerigid.mha'):
    """
    Creates a deformation field by scaling a 3x3 transformation matrix and 
    composing it with given vector fields.

    Parameters
    ----------
    shape_transform : np.ndarray
        A 3x3 transformation matrix (affine or rigid).
    source_prep : np.ndarray
        Source coordinate grid or image array used for transformation.
    u_x, u_y : np.ndarray
        Components of an existing deformation field (x and y displacement maps).
    util : module
        A module or object providing two functions:
            - matrix_df(source, transform): applies rigid transformation.
            - combine_deformation(u_x, u_y, t_x, t_y): composes deformation fields.
    output_path : str, optional
        Path to save the resulting deformation field as an .mha file.

    Returns
    -------
    sitk.Image
        The resulting deformation field as a SimpleITK image.
    """

    # Define scale factor
    scale_factor = 0.625 / 40  # equivalent to 0.015625

    # Scale the translation components of the transformation matrix
    transform_3x3_scaled = shape_transform.copy()
    transform_3x3_scaled[0:2, 2] *= scale_factor

    # Apply inverse of scaled transform
    sh_transform = transform_3x3_scaled
    t_x, t_y = matrix_df(source_prep, np.linalg.inv(sh_transform))

    # Compose the new vector field
    f_x, f_y = combine_deformation(u_x, u_y, t_x, t_y)

    # Stack into deformation field
    deformation_field = np.stack((f_x, f_y), axis=-1)

    # Convert to SimpleITK image and save
    sitk_image = sitk.GetImageFromArray(deformation_field)
    sitk.WriteImage(sitk_image, output_path)

    return sitk_image
def create_nonrigid_mha(
    moving_subsample,
    nonrigid_transformed_coords,
    r_x,
    w_x, w_y,
    target_prep,
    create_displacement_field,
    RegistrationParams,
    output_path=""
):
    """
    Builds a non-rigid displacement + deformation field 
    and saves it as an .mha file (using fr_x, fr_y as final field).
    """

    print("Creating displacement field...")

    # Scale for numerical stability
    scale_factor = 64
    source_points_scaled = moving_subsample / scale_factor
    target_points_scaled = nonrigid_transformed_coords / scale_factor

    # Determine grid size
    H, W = r_x.shape
    grid_y, grid_x = np.mgrid[0:H, 0:W]

    # Dense displacement field
    displacement_field = create_displacement_field(
        source_points_scaled,
        target_points_scaled,
        target_prep.shape,
        method=RegistrationParams.INTERPOLATION_METHOD,
        sigma=RegistrationParams.DISPLACEMENT_SIGMA,
        max_displacement=RegistrationParams.MAX_DISPLACEMENT
    )

    # Combine deformation + displacement
    fr_x, fr_y = util.combine_deformation(
        w_x, w_y,
        displacement_field[..., 0],
        displacement_field[..., 1]
    )

    # FINAL deformation field to save (now fr_x, fr_y)
    deformation_field = np.stack((fr_x, fr_y), axis=-1)

    # Convert → SimpleITK image
    sitk_image = sitk.GetImageFromArray(deformation_field)

    # Save to disk
    sitk.WriteImage(sitk_image, output_path)

    return sitk_image, deformation_field, displacement_field, (fr_x, fr_y)

def create_deform(source_prep, final_transform, displacement_field, output_path=""):
    """
    Computes the full deformation field and writes it to an MHA file.

    Parameters
    ----------
    source_prep : array-like
        Input source data for matrix_df.
    final_transform : array-like
        Final transform data for matrix_df.
    displacement_field : array-like
        Raw displacement field (expected shape: [2, H, W] or similar).
    output_path : str
        Output MHA file path.
    """
    
    # Get rigid or base transform components
    r_x, r_y = matrix_df(source_prep, final_transform)

    # Convert displacement field into 2-component arrays
    disp_field = deform_conversion(displacement_field)

    # Combine base transform + deformation
    w_x, w_y = combine_deformation(r_x, r_y, disp_field[0], disp_field[1])

    # Stack into deformation field (H, W, 2)
    deformation_field = np.stack((w_x, w_y), axis=-1)

    # Convert to SimpleITK image
    sitk_image = sitk.GetImageFromArray(deformation_field)

    # Write to file
    sitk.WriteImage(sitk_image, output_path)

    return sitk_image
def apply_deformation_to_points(points, deformation_field):
    """
    Apply a deformation field to a set of 2D points.

    Args:
        points: np.array of shape (N, 2), points as (x, y) coordinates.
        deformation_field: np.array of shape (2, H, W), displacement vectors.
                           deformation_field[0] is displacement in x,
                           deformation_field[1] is displacement in y.

    Returns:
        warped_points: np.array of shape (N, 2), warped point coordinates.
    """
    # Extract deformation components
    disp_x = deformation_field[0]
    disp_y = deformation_field[1]

    H, W = disp_x.shape
    # Interpolate displacement at each point's coordinate
    warped_points = np.zeros_like(points)

    # Points are (x, y), but map_coordinates requires (row, col) = (y, x)
    # So we need to query displacement fields at (y, x)
    coords = np.array([points[:,1], points[:,0]])  # shape (2, N)

    # Interpolate displacement fields at these coords
    disp_at_points_x = map_coordinates(disp_x, coords, order=1, mode='nearest')
    disp_at_points_y = map_coordinates(disp_y, coords, order=1, mode='nearest')

    # Apply displacement: new_pos = original_pos + displacement
    warped_points[:, 0] = points[:, 0] + disp_at_points_x
    warped_points[:, 1] = points[:, 1] + disp_at_points_y

    return warped_points
def create_displacement_field_for_wsi(transform_matrix, source_thumbnail, target_thumbnail):    
    # Use the larger dimensions to avoid cropping issues
    max_height = max(source_thumbnail.shape[0], target_thumbnail.shape[0])
    max_width = max(source_thumbnail.shape[1], target_thumbnail.shape[1])
    
    # Create coordinate grid at the chosen dimensions
    y_coords, x_coords = np.mgrid[0:max_height, 0:max_width]
    
    # Convert to homogeneous coordinates for transformation
    coords_homogeneous = np.stack([
        x_coords.flatten(), 
        y_coords.flatten(), 
        np.ones(x_coords.size)
    ], axis=0)
    
    transform_inv = np.linalg.inv(transform_matrix)
    transformed_coords = transform_inv @ coords_homogeneous
    
    # Reshape back to 2D grids
    source_x = transformed_coords[0].reshape(max_height, max_width)
    source_y = transformed_coords[1].reshape(max_height, max_width)
    
    u_x = source_x - x_coords
    u_y = source_y - y_coords
    # Stack into displacement field (H, W, 2)
    displacement_field = np.stack((u_x, u_y), axis=-1)
    return displacement_field
# POINT


def _check_dims(
    fixed_img: np.ndarray,
    moving_img: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Check the dimensionality of images and mask.

    This function verify the dimensionality of images and their corresponding masks.
    If the input images are RGB images, it converts them to grayscale images.

    Args:
        fixed_img (:class:`numpy.ndarray`):
            A fixed image.
        moving_img (:class:`numpy.ndarray`):
            A moving image.
        fixed_mask (:class:`numpy.ndarray`):
            A binary tissue mask for the fixed image.
        moving_mask (:class:`numpy.ndarray`):
            A binary tissue mask for the moving image.

    Returns:
        tuple:
            - :class:`numpy.ndarray` - A grayscale fixed image.
            - :class:`numpy.ndarray` - A grayscale moving image.

    """
    if len(np.unique(fixed_mask)) == 1 or len(np.unique(moving_mask)) == 1:
        msg = "The foreground is missing in the mask."
        raise ValueError(msg)

    if (
        fixed_img.shape[:2] != fixed_mask.shape
        or moving_img.shape[:2] != moving_mask.shape
    ):
        msg = "Mismatch of shape between image and its corresponding mask."
        raise ValueError(msg)

    if len(fixed_img.shape) == RGB_IMAGE_DIM:
        fixed_img = cv2.cvtColor(fixed_img, cv2.COLOR_BGR2GRAY)

    if len(moving_img.shape) == RGB_IMAGE_DIM:
        moving_img = cv2.cvtColor(moving_img, cv2.COLOR_BGR2GRAY)

    return fixed_img, moving_img



def compute_center_of_mass(mask: np.ndarray) -> tuple:
    """Compute center of mass.

    Args:
        mask: (:class:`numpy.ndarray`):
            A binary mask.

    Returns:
        :py:obj:`tuple` - x- and y- coordinates representing center of mass.
            - :py:obj:`int` - X coordinate.
            - :py:obj:`int` - Y coordinate.

    """
    moments = cv2.moments(mask)
    x_coord_center = moments["m10"] / moments["m00"]
    y_coord_center = moments["m01"] / moments["m00"]
    return (x_coord_center, y_coord_center)


def apply_affine_transformation(
    fixed_img: np.ndarray,
    moving_img: np.ndarray,
    transform_initializer: np.ndarray,
) -> np.ndarray:
    """Apply affine transformation using OpenCV.

    Args:
        fixed_img (:class:`numpy.ndarray`):
            A fixed image.
        moving_img (:class:`numpy.ndarray`):
            A moving image.
        transform_initializer (:class:`numpy.ndarray`):
            A rigid transformation matrix.

    Returns:
        :class:`numpy.ndarray`:
            A transformed image.

    Examples:
        >>> moving_image = apply_affine_transformation(
        ...     fixed_image, moving_image, transform_initializer
        ... )

    """
    return cv2.warpAffine(
        moving_img,
        transform_initializer[0:-1][:],
        fixed_img.shape[:2][::-1],
    )
# checkpoint conversion
def convert_pytorch_checkpoint(net_state_dict):
    variable_name_list = list(net_state_dict.keys())
    is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
    if is_in_parallel_mode:
        net_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
        }
    return net_state_dict

def create_pyramid(array: np.ndarray, num_levels: int, mode: str = 'bilinear'):
    """
    Creates a resolution pyramid for a NumPy array (assumes uniform downsampling by a factor of 2).

    Parameters
    ----------
    array : np.ndarray
        The input array (image or multi-dimensional data).
    num_levels : int
        The number of pyramid levels.
    mode : str
        The interpolation mode ("bilinear" or "nearest").

    Returns
    ----------
    pyramid : list of np.ndarray
        The created resolution pyramid.
    """
    pyramid = [None] * num_levels
    for i in range(num_levels - 1, -1, -1):
        if i == num_levels - 1:
            pyramid[i] = array
        else:
            current_size = pyramid[i + 1].shape
            new_size = tuple(int(current_size[j] / 2) if j > 1 else current_size[j] for j in range(len(current_size)))
            new_size = new_size[2:]  # Exclude batch and channel dimensions
            new_array = resample_tensor_to_size(gaussian_smoothing(pyramid[i + 1], 1), new_size, mode=mode)
            pyramid[i] = new_array
    return pyramid

def scale_transformation_matrix(transform_matrix, input_res, output_res):
    """
    Upscale the transformation matrix to the original image resolution.

    Args:
        transform_matrix (np.ndarray): Transformation matrix.
        resolution (int): Original image resolution.

    Returns:
        np.ndarray: Upscaled transformation matrix.
    """
    scale_factor = output_res / input_res
    transform_upscaled = transform_matrix.copy()
    transform_upscaled[0:2][:, 2] = transform_upscaled[0:2][:, 2] * scale_factor

    return transform_upscaled

def warp_image(image, u_x, u_y):
    y_size, x_size = image.shape
    grid_x, grid_y = np.meshgrid(np.arange(x_size), np.arange(y_size))
    return nd.map_coordinates(image, [grid_y + u_y, grid_x + u_x], order=3, cval=0.0)


def matrix_mha(image, matrix):
    y_size, x_size = np.shape(image)
    x_grid, y_grid = np.meshgrid(np.arange(x_size), np.arange(y_size))
    points = np.vstack((x_grid.ravel(), y_grid.ravel(), np.ones(np.shape(image)).ravel()))
    transformed_points = matrix @ points
    u_x = np.reshape(transformed_points[0, :], (y_size, x_size)) - x_grid
    u_y = np.reshape(transformed_points[1, :], (y_size, x_size)) - y_grid
    return u_x, u_y

def combine_deformation(u_x, u_y, v_x, v_y):
    y_size, x_size = np.shape(u_x)
    grid_x, grid_y = np.meshgrid(np.arange(x_size), np.arange(y_size))
    added_y = grid_y + v_y
    added_x = grid_x + v_x
    t_x = nd.map_coordinates(grid_x + u_x, [added_y, added_x], mode='constant', cval=0.0)
    t_y = nd.map_coordinates(grid_y + u_y, [added_y, added_x], mode='constant', cval=0.0)
    n_x, n_y = t_x - grid_x, t_y - grid_y
    indexes_x = np.logical_or(added_x >= x_size - 1, added_x <= 0)
    indexes_y = np.logical_or(added_y >= y_size - 1, added_y <= 0)
    indexes = np.logical_or(indexes_x, indexes_y)
    n_x[indexes] = 0.0
    n_y[indexes] = 0.0
    return n_x, n_y

def apply_displacement_field(image, displacement_field):
    """
    Apply displacement field to warp an image.
    
    Args:
        image: Input image
        displacement_field: Displacement field [2, H, W]
        
    Returns:
        warped_image: Warped image
    """
    h, w = image.shape[:2]
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    
    # Add displacement to grid coordinates
    coords_y = grid_y + displacement_field[1]
    coords_x = grid_x + displacement_field[0]
    
    # Stack coordinates for mapping
    coords = np.stack([coords_y, coords_x], axis=0)
    
    # Apply mapping to each channel
    if len(image.shape) == 3:
        warped_image = np.zeros_like(image)
        for c in range(image.shape[2]):
            warped_image[:, :, c] = nd.map_coordinates(image[:, :, c], coords, order=1)
    else:
        warped_image = nd.map_coordinates(image, coords, order=1)
    
    return warped_image

# ------------------------
def sort_coordinates(set1, set2):
    """
    Match points between fixed and moving point set.

    Args:
        set1 (np.array): Fixed point set
        set2 (np.array): Moving point set

    Returns:
        np.array: New moving point set
    """
    # Match points using K nearest neighbors.
    sorted_move = []
    set1 = np.array([[coord[0], coord[1]] for coord in set1])
    set2 = np.array([[coord[0], coord[1]] for coord in set2])
    knn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(set2)
    for coord in set1:
        distances, indices = knn.kneighbors([[coord[0], coord[1]]])
        sorted_move.append(set2[indices])
    # Return matched points.
    return np.array(sorted_move)

def dice(fixed, moving):
    # Calculate DICE coefficient between fixed and moving images.
    intersection = np.sum(np.logical_and(fixed, moving))
    union = np.sum(np.logical_or(fixed, moving))
    
    if union == 0:
        return 0  # handle the case where both arrays are empty
    
    dice_coefficient = 2.0 * intersection / union
    return dice_coefficient

def mse(fixed, moving):
    """
    Calculate MSE value.

    Args:
        fixed (np.array): Fixed point set
        moving (np.array): Moving point set

    Returns:
        float: Mean Square error.
    """
    if len(fixed) == 0 or len(moving) == 0:
        print("Warning: One or both point sets are empty. Returning inf for MSE.")
        return float('inf') 
    else:
        # Match fixed and moving point set.
        moving = sort_coordinates(fixed, moving)
        s = 0
        # Calculate euclidean distances between matched points.
        for f, m in zip(fixed, moving):
            s += (f[0]-m[0][0][0])**2 + (f[1]-m[0][0][1])**2
        # Calculates MSE value and return value
        mse_value = s/len(fixed)

    return mse_value



def rotate_point(x, y, cx, cy, angle):
    """
    Rotate point set around center with certain angle.

    Args:
        x (float): x-coordinate
        y (float): y-coordinate
        cx (float): center x-coordinate
        cy (float): center y-coordinate
        angle (float): rotation angle

    Returns:
        float: rotated x coordinate
        float: rotated y coordinate
    """
    # convert angles to radians
    angle_radians = math.radians(angle)
    
    # Apply rotation formula
    x_prime = (x - cx) * math.cos(angle_radians) - (y - cy) * math.sin(angle_radians) + cx
    y_prime = (x - cx) * math.sin(angle_radians) + (y - cy) * math.cos(angle_radians) + cy
    
    return x_prime, y_prime


    
def phase_correlation(fixed, moving):
    # Run phase correlation shift between point sets as second measure for translation.
    shift, _, _ = phase_cross_correlation(fixed, moving)
    return np.array(shift)


def find_scale(images):
    # Find x and y scale factors between fixed and moving point set using the difference between min and max x and y point sets.
    min_x_move, max_x_move, min_y_move, max_y_move = np.min(images.moving.points[:, 0]), np.max(images.moving.points[:, 0]), np.min(images.moving.points[:, 1]), np.max(images.moving.points[:, 1])
    min_x_fixed, max_x_fixed, min_y_fixed, max_y_fixed = np.min(images.fixed.points[:, 0]), np.max(images.fixed.points[:, 0]), np.min(images.fixed.points[:, 1]), np.max(images.fixed.points[:, 1])
    scale_factor_x = (max_x_fixed-min_x_fixed) / (max_x_move-min_x_move)
    scale_factor_y = (max_y_fixed-min_y_fixed) / (max_y_move-min_y_move)
    return scale_factor_x, scale_factor_y

def scale_coordinates(images):
    # Find scale factors and apply onto moving point sets.
    scale_factor_x , scale_factor_y = find_scale(images)
    scale_factor = max(scale_factor_x, scale_factor_y)
    min_x_move, max_x_move, min_y_move, max_y_move = np.min(images.moving.points[:, 0]), np.max(images.moving.points[:, 0]), np.min(images.moving.points[:, 1]), np.max(images.moving.points[:, 1])
    x_center = (max_x_move+min_x_move) / 2
    y_center = (max_y_move+min_y_move) / 2
    translated_coordinates = [(x - x_center, y - y_center) for x, y in images.moving.points]

    scaled_coordinates = [(scale_factor_x * x, scale_factor_y * y) for x, y in translated_coordinates]

    final_coordinates = np.array([[x + x_center, y + y_center] for x, y in scaled_coordinates])
    # Apply scalings and set point sets.
    images.moving.set_points(final_coordinates)
    return images, 1/scale_factor_x, 1/scale_factor_y


def matchpoints(points):
    """
    Match point between fixed and moving point sets.

    Args:
        points (np.array): point sets

    Returns:
        np.array: Rotation matrix and translation vector
    """
    if not points:
        return None, None

    points = np.array(points)

    means = np.mean(points, axis=0)
    deviations = points - means
    # Match point set between closest points in ICP point sets.
    s_x_xp = np.sum(deviations[:, 0, 0] * deviations[:, 1, 0])
    s_y_yp = np.sum(deviations[:, 0, 1] * deviations[:, 1, 1])
    s_x_yp = np.sum(deviations[:, 0, 0] * deviations[:, 1, 1])
    s_y_xp = np.sum(deviations[:, 0, 1] * deviations[:, 1, 0])
    # Calculate rotation matrix between matched points
    rot_angle = np.arctan2(s_x_yp - s_y_xp, s_x_xp + s_y_yp)
    # Calculate translation vector between matched points
    translation = np.array([
        means[1, 0] - (means[0, 0] * np.cos(rot_angle) - means[0, 1] * np.sin(rot_angle)),
        means[1, 1] - (means[0, 0] * np.sin(rot_angle) + means[0, 1] * np.cos(rot_angle))
    ])
    # Return transformation.
    return  np.array([[math.cos(rot_angle), -math.sin(rot_angle)],
                        [math.sin(rot_angle), math.cos(rot_angle)]]), translation
    
def resample(image, output_x_size, output_y_size):
    y_size, x_size = np.shape(image)
    out_grid_x, out_grid_y = np.meshgrid(np.arange(output_x_size), np.arange(output_y_size))
    out_grid_x = out_grid_x * x_size / output_x_size
    out_grid_y = out_grid_y * y_size / output_y_size
    image = nd.map_coordinates(image, [out_grid_y, out_grid_x], order=3, cval=0.0)
    return image

def resample_both(source, target, resample_ratio):
    s_y_size, s_x_size = source.shape
    t_y_size, t_x_size = target.shape
    source = resample(source, int(s_x_size/resample_ratio), int(s_y_size/resample_ratio))
    target = resample(target, int(t_x_size/resample_ratio), int(t_y_size/resample_ratio))
    return source, target

def resample_displacement_field(u_x, u_y, output_x_size, output_y_size):
    y_size, x_size = np.shape(u_x)
    u_x = resample(u_x, output_x_size, output_y_size)
    u_y = resample(u_y, output_x_size, output_y_size)
    u_x = u_x * output_x_size/x_size
    u_y = u_y * output_y_size/y_size
    return u_x, u_y

def matrix_df(image, matrix):
    y_size, x_size, _ = np.shape(image)
    x_grid, y_grid = np.meshgrid(np.arange(x_size), np.arange(y_size))
    points = np.vstack((x_grid.ravel(), y_grid.ravel(), np.ones_like(x_grid).ravel()))
    transformed_points = matrix @ points
    u_x = np.reshape(transformed_points[0, :], (y_size, x_size)) - x_grid
    u_y = np.reshape(transformed_points[1, :], (y_size, x_size)) - y_grid
    return u_x, u_y
    
def load_image_as_grayscale(path):
    """Load an image and convert it to grayscale."""
    image = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(image)
    if image.ndim == 4:  # [z, y, x, c] format
        image = image[:, :, :3]
    image = color.rgb2gray(image)
    return image

def load_landmarks_no_header(path):
    """
    Load the first two columns of landmark data from a CSV file (no header).
    """
    landmarks = pd.read_csv(path, header=None, usecols=[0, 1]).values.astype(np.float64)
    return landmarks


def save_landmarks(path, landmarks):
    df = pd.DataFrame(landmarks, columns=['X', 'Y'])
    df.index = np.arange(1, len(df) + 1)
    df.to_csv(path)

def pad_landmarks(landmarks, x, y):
    landmarks[:, 0] += x
    landmarks[:, 1] += y
    return landmarks

def plot_landmarks(landmarks, marker_type, colors=None):
    landmarks_length = len(landmarks)
    if colors is None:
        colors = np.random.uniform(0, 1, (3, landmarks_length))
    for i in range(landmarks_length):
        plt.plot(landmarks[i, 0], landmarks[i, 1], marker_type, color=colors[:, i])
    return colors

def normalize(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def to_image(array):
    return sitk.GetImageFromArray((255*array).astype(np.uint8))

def calculate_resample_size(source, target, output_max_size):
    target_y_size, target_x_size = np.shape(target)[0:2]
    source_y_size, source_x_size = np.shape(source)[0:2]

    max_y_size = max(source_y_size, target_y_size)
    max_x_size = max(source_x_size, target_x_size)

    max_dim = max(max_y_size, max_x_size)
    rescale_ratio = max_dim/output_max_size
    return rescale_ratio



def deform_conversion(displacement_field_tc: torch.Tensor):
    ndim = len(displacement_field_tc.size()) - 2
    if ndim == 2:
        displacement_field_np = displacement_field_tc.detach().clone().cpu()[0].permute(2, 0, 1).numpy()
        shape = displacement_field_np.shape
        temp_df_copy = displacement_field_np.copy()
        displacement_field_np[0, :, :] = temp_df_copy[0, :, :] / 2.0 * (shape[2])
        displacement_field_np[1, :, :] = temp_df_copy[1, :, :] / 2.0 * (shape[1])
    elif ndim == 3:
        displacement_field_np = displacement_field_tc.detach().clone().cpu()[0].permute(3, 0, 1, 2).numpy()
        shape = displacement_field_np.shape
        temp_df_copy = displacement_field_np.copy()
        displacement_field_np[0, :, :, :] = temp_df_copy[1, :, :, :] / 2.0 * (shape[2])
        displacement_field_np[1, :, :, :] = temp_df_copy[2, :, :, :] / 2.0 * (shape[1])
        displacement_field_np[2, :, :, :] = temp_df_copy[0, :, :, :] / 2.0 * (shape[3])
    return displacement_field_np

def gaussian_filter(image, sigma):
    return nd.gaussian_filter(image, sigma)

def round_up_to_odd(value):
    return int(np.ceil(value) // 2 * 2 + 1)


def load_image(path):
    """
    Load an image and return it as RGB (not converting to grayscale).
    """
    image = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(image)
    
    # If image is grayscale but has a singleton dimension for color, squeeze it
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = np.squeeze(image)
    
    # Ensure RGB format for color images
    if len(image.shape) == 2:
        # Convert grayscale to RGB
        image = np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3 and image.shape[2] > 3:
        # Take only first 3 channels if there are more (e.g., RGBA)
        image = image[:, :, :3]
        
    return image


def load_landmarks(path):
    """
    Load landmarks from a CSV file.
    """
    landmarks = pd.read_csv(path).iloc[:, 1:].values.astype(np.float64)
    return landmarks


def transform_landmarks(landmarks, u_x, u_y):
    landmarks_x = landmarks[:, 0]
    landmarks_y = landmarks[:, 1]
    ux = nd.map_coordinates(u_x, [landmarks_y, landmarks_x], mode='nearest')
    uy = nd.map_coordinates(u_y, [landmarks_y, landmarks_x], mode='nearest')
    new_landmarks = np.stack((landmarks_x + ux, landmarks_y + uy), axis=1)
    return new_landmarks

def tre(landmarks_1, landmarks_2):
    tre = np.sqrt(np.square(landmarks_1[:, 0] - landmarks_2[:, 0]) + np.square(landmarks_1[:, 1] - landmarks_2[:, 1]))
    return tre

def rtre(landmarks_1, landmarks_2, x_size, y_size):
    return tre(landmarks_1, landmarks_2) / np.sqrt(x_size*x_size + y_size*y_size)

def print_rtre(source_landmarks, target_landmarks, x_size, y_size):
    calculated_tre = tre(source_landmarks, target_landmarks)
    mean = np.mean(calculated_tre) * 100
    median = np.median(calculated_tre) * 100
    mmax = np.max(calculated_tre) * 100
    mmin = np.min(calculated_tre) * 100
    print("TRE mean [%]: ", mean)
    print("TRE median [%]: ", median)
    print("TRE max [%]: ", mmax)
    print("TRE min [%]: ", mmin)
    return mean, median, mmax, mmin


def compute_center_of_mass(array):
    # Calculates the center of mass of point set.
    center_of_mass_2d = np.mean(array, axis=0)[:-1]
    return center_of_mass_2d
