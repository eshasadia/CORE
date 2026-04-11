"""
Preprocessing functions for WSI registration
"""
import numpy as np
import cv2
import math
from tiatoolbox.wsicore.wsireader import WSIReader, VirtualWSIReader
import core.preprocessing.tissuemask as tissuemask
import core.preprocessing.preprocessing as prep  # Original preprocessing module
def load_wsi_images(source_path, target_path, resolution=0.625, data='', obj_power=''):
    """
    Load source and target WSI images

    Args:
        source_path: Path to source WSI
        target_path: Path to target WSI
        resolution: Resolution for loading
        data: Dataset type (e.g. 'anhir')
        obj_power: Objective power for VirtualWSIReader

    Returns:
        tuple: (source_wsi, target_wsi, source_image, target_image)
    """
    if data == 'anhir':
        target_wsi = VirtualWSIReader.open(target_path)
        source_wsi = VirtualWSIReader.open(source_path)

        # Manually set the objective power if it's None
        if target_wsi.info.objective_power is None:
            target_wsi.info.objective_power = obj_power # Set to a standard value like 20x or 40x
        if source_wsi.info.objective_power is None:
            source_wsi.info.objective_power = obj_power
        
        target = target_wsi.slide_thumbnail(resolution=resolution, units="power")
        source = source_wsi.slide_thumbnail(resolution=resolution, units="power")
        
    else:
        source_wsi = WSIReader.open(source_path)
        target_wsi = WSIReader.open(target_path)
        
        # Load images at specified resolution
        source = load_slide(source_path, resolution)
        target = load_slide(target_path, resolution)
        
        print(f"Source original shape: {source.shape}")
        print(f"Target original shape: {target.shape}")
    
    return source_wsi, target_wsi, source, target


def preprocess_images(source, target):
    """
    Preprocess source and target images
    
    Args:
        source: Source image array
        target: Target image array
        
    Returns:
        tuple: (source_prep, target_prep)
    """
 
    source_prep, target_prep = source, target
    
    print(f"Source preprocessed shape: {source_prep.shape}")
    print(f"Target preprocessed shape: {target_prep.shape}")
    
    return source_prep, target_prep


def extract_tissue_masks(source_prep, target_prep, artefacts):
    """
    Extract tissue masks from preprocessed images
    
    Args:
        source_prep: Preprocessed source image
        target_prep: Preprocessed target image
        
    Returns:
        tuple: (source_mask, target_mask)
    """
    extractor = tissuemask.FlorenceTissueMaskExtractor()
    source_mask = extractor.extract(source_prep, artefacts)
    target_mask = extractor.extract(target_prep, artefacts)
    
    return source_mask, target_mask


def adjust_gamma(image, gamma=0.5):
    """
    Apply gamma correction to an image
    
    Args:
        image: Input image
        gamma: Gamma value
        
    Returns:
        Gamma corrected image
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                     for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def scale_transformation_matrix(transform_matrix, input_res, output_res):
    """
    Scale transformation matrix to different resolution
    
    Args:
        transform_matrix: Input transformation matrix
        input_res: Input resolution
        output_res: Output resolution
        
    Returns:
        Scaled transformation matrix
    """
    scale_factor = output_res / input_res
    transform_scaled = transform_matrix.copy()
    transform_scaled[0:2, 2] = transform_scaled[0:2, 2] * scale_factor
    return transform_scaled



def load_slide(image_path_1: str, resolution: float = 0.625):
    fixed_wsi = WSIReader.open(image_path_1)
    fixed_thumbnail = fixed_wsi.slide_thumbnail(resolution=resolution, units="power")

    return fixed_thumbnail


def pad_single(image, new_shape):
    y_size, x_size, _ = image.shape
    y_pad = (int(np.floor((new_shape[0] - y_size) / 2)), int(np.ceil((new_shape[0] - y_size) / 2)))
    x_pad = (int(np.floor((new_shape[1] - x_size) / 2)), int(np.ceil((new_shape[1] - x_size) / 2)))
    new_image = np.pad(image, (y_pad, x_pad, (0, 0)), constant_values=0)
    return new_image


def pad_images_np(source, target):
    y_size_source, x_size_source, _ = source.shape
    y_size_target, x_size_target, _ = target.shape
    new_y_size = max(y_size_source, y_size_target)
    new_x_size = max(x_size_source, x_size_target)
    new_shape = (new_y_size, new_x_size)

    padded_source = pad_single(source, new_shape)
    padded_target = pad_single(target, new_shape)
    return padded_source, padded_target

def pad_to_same_size(image_1: np.ndarray, image_2: np.ndarray, pad_value: float = 1.0):
    """
    Pad two images to the same size.
    
    Args:
        image_1: First image array with shape (height, width, channel), (batch, height, width, channel), 
                 (height, width), or (batch, channel, height, width)
        image_2: Second image array with shape (height, width, channel), (batch, height, width, channel), 
                 (height, width), or (batch, channel, height, width)
        pad_value: Value to use for padding
        
    Returns:
        tuple: (padded_image_1, padded_image_2, padding_params)
    """
    # Determine the dimensionality and shape of each image
    if image_1.ndim == 4:  # (batch, channel, height, width) format
        y_size_1, x_size_1 = image_1.shape[2], image_1.shape[3]
        y_size_2, x_size_2 = image_2.shape[2], image_2.shape[3]
    elif image_1.ndim == 3:  # (height, width, channel) format - typical RGB image
        y_size_1, x_size_1 = image_1.shape[0], image_1.shape[1]
        y_size_2, x_size_2 = image_2.shape[0], image_2.shape[1]
    else:  # (height, width) format - grayscale image
        y_size_1, x_size_1 = image_1.shape
        y_size_2, x_size_2 = image_2.shape
    
    pad_1 = [(0, 0), (0, 0)]
    pad_2 = [(0, 0), (0, 0)]
    
    # Handle height padding
    if y_size_1 > y_size_2:
        pad_size = y_size_1 - y_size_2
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_2[0] = pad
    elif y_size_1 < y_size_2:
        pad_size = y_size_2 - y_size_1
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_1[0] = pad
    
    # Handle width padding
    if x_size_1 > x_size_2:
        pad_size = x_size_1 - x_size_2
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_2[1] = pad
    elif x_size_1 < x_size_2:
        pad_size = x_size_2 - x_size_1
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_1[1] = pad
    
    # Create padded arrays based on dimensionality
    if image_1.ndim == 4:  # (batch, channel, height, width)
        # For 4D tensors: add padding for height and width dimensions
        padded_image_1 = np.pad(
            image_1,
            ((0, 0), (0, 0), pad_1[0], pad_1[1]),
            mode='constant',
            constant_values=pad_value
        )
        padded_image_2 = np.pad(
            image_2,
            ((0, 0), (0, 0), pad_2[0], pad_2[1]),
            mode='constant',
            constant_values=pad_value
        )
    elif image_1.ndim == 3:  # (height, width, channel) - RGB images
        # For RGB images: pad height and width, preserve channels
        padded_image_1 = np.pad(
            image_1,
            (pad_1[0], pad_1[1], (0, 0)),
            mode='constant',
            constant_values=pad_value
        )
        padded_image_2 = np.pad(
            image_2,
            (pad_2[0], pad_2[1], (0, 0)),
            mode='constant',
            constant_values=pad_value
        )
    else:  # (height, width) - grayscale images
        # For 2D arrays
        padded_image_1 = np.pad(
            image_1,
            pad_1,
            mode='constant',
            constant_values=pad_value
        )
        padded_image_2 = np.pad(
            image_2,
            pad_2,
            mode='constant',
            constant_values=pad_value
        )
    
    padding_params = {
        'pad_1': pad_1,
        'pad_2': pad_2
    }
    return padded_image_1, padded_image_2, padding_params


def resize_and_compute_translation(moving_image, fixed_image):
    """
    Resizes fixed and moving images to the maximum dimensions using black padding
    and computes initial translation offsets for 2D or 3D images (where the 3rd dimension is the channel).

    Args:
        fixed_image (np.ndarray): The fixed image (2D or 3D).
        moving_image (np.ndarray): The moving image (2D or 3D).

    Returns:
        fixed_padded (np.ndarray): The fixed image with padding.
        moving_padded (np.ndarray): The moving image with padding.
        translation (tuple): The translation offsets (tx, ty) for 2D or (tx, ty, channels) for 3D.
    """
    # Ensure input images are in uint8 format (scaling to 0-255 range if necessary)
    if fixed_image.dtype != np.uint8:
        fixed_image = (fixed_image * 255).astype(np.uint8)
    if moving_image.dtype != np.uint8:
        moving_image = (moving_image * 255).astype(np.uint8)
    
    # Check for 2D or 3D images (where 3D is understood as height, width, channels)
    if fixed_image.ndim == 2:  # 2D images
        fixed_h, fixed_w = fixed_image.shape
        moving_h, moving_w = moving_image.shape
        
        # Compute padding and translation offsets
        max_h, max_w = max(fixed_h, moving_h), max(fixed_w, moving_w)
        if moving_h > fixed_h:
            tx, ty = (max_w - moving_w) // 2, (max_h - moving_h) // 2
        fx, fy = max_w - fixed_w, max_h - fixed_h
        mx, my = max_w - moving_w, max_h - moving_h
        
        # Padding images to match max dimensions
        fixed_padded = np.zeros((max_h, max_w), dtype=np.uint8)
        moving_padded = np.zeros((max_h, max_w), dtype=np.uint8)
        fixed_padded[:fixed_h, :fixed_w] = fixed_image
        # moving_padded[ty:ty + moving_h, tx:tx + moving_w] = moving_image
        moving_padded[:moving_h, :moving_w] = moving_image
        
        return fixed_padded, moving_padded, (fx, fy), (mx, my)
    
    elif fixed_image.ndim == 3:  # 3D images (height, width, channels)
        fixed_h, fixed_w, fixed_c = fixed_image.shape
        moving_h, moving_w, moving_c = moving_image.shape
        
        # Compute padding and translation offsets for height, width (not channels)
        max_h, max_w = max(fixed_h, moving_h), max(fixed_w, moving_w)
        tx, ty = (max_w - moving_w) // 2, (max_h - moving_h) // 2
        fx, fy = max_w - fixed_w, max_h - fixed_h
        mx, my = max_w - moving_w, max_h - moving_h
        
        # Padding images to match max height and width (but retain the channel dimension)
        fixed_padded = np.zeros((max_h, max_w, fixed_c), dtype=np.uint8)
        moving_padded = np.zeros((max_h, max_w, moving_c), dtype=np.uint8)
        moving_padded[:moving_h, :moving_w] = moving_image
        fixed_padded[:fixed_h, :fixed_w, :] = fixed_image
        # moving_padded[ty:ty + moving_h, tx:tx + moving_w, :] = moving_image
        fixed_padded = gamma_corrections(fixed_padded, 1)
        moving_padded = gamma_corrections(moving_padded, 0.4)
        return fixed_padded, moving_padded, (fx, fy), (mx, my)
    
    else:
        raise ValueError("Input images must be either 2D or 3D with channel as the 3rd dimension.")


def gamma_corrections(img, gamma):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)
def process_nuclei_patch(img, threshold, gamma=None, min_area=200):
    """
    Process a single patch to detect nuclei
    
    Args:
        img: Input image patch
        threshold: Binary threshold value
        gamma: Gamma correction value (optional)
        min_area: Minimum area for nuclei detection
        
    Returns:
        tuple: (binary_image, stats, centroids)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply gamma correction if specified
    if gamma is not None:
        gray = adjust_gamma(gray, gamma)
    
    # Apply binary threshold
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    
    return binary, stats, centroids
