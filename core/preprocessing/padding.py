from typing import Dict, Iterable, List, Tuple
import math
import numpy as np


def calculate_pad_value(size_1: Iterable[int], size_2: Iterable[int]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Calculates the padding required to make two images the same size."""
    y_size_1, x_size_1 = size_1
    y_size_2, x_size_2 = size_2
    pad_1 = [(0, 0), (0, 0)]
    pad_2 = [(0, 0), (0, 0)]
    
    if y_size_1 > y_size_2:
        pad_size = y_size_1 - y_size_2
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_2[0] = pad
    elif y_size_1 < y_size_2:
        pad_size = y_size_2 - y_size_1
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_1[0] = pad
        
    if x_size_1 > x_size_2:
        pad_size = x_size_1 - x_size_2
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_2[1] = pad
    elif x_size_1 < x_size_2:
        pad_size = x_size_2 - x_size_1
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_1[1] = pad
        
    return pad_1, pad_2

def apply_padding_landmarks(landmarks: np.ndarray, pad: List[Tuple[int, int]]) -> np.ndarray:
    """Applies padding to landmark coordinates."""
    padded_landmarks = landmarks.copy()
    padded_landmarks[:, 0] += pad[1][0]  # x coordinate
    padded_landmarks[:, 1] += pad[0][0]  # y coordinate
    return padded_landmarks



def pad_image_src(
    source: np.ndarray,
    target: np.ndarray,
    pad_value: int = 255  # white padding for uint8
) -> Tuple[np.ndarray, np.ndarray, Dict, np.ndarray, np.ndarray]:
    """
    Pads or crops the source image so it matches the target's size.
    Target remains unchanged.

    Positive padding if source < target.
    Negative padding (crop) if source > target.
    """

    src_h, src_w = source.shape[:2]
    tgt_h, tgt_w = target.shape[:2]

    # Compute differences
    dy = tgt_h - src_h
    dx = tgt_w - src_w

    pad_src = [[0, 0], [0, 0]]  # store applied padding/cropping

    # --- Y-axis (height) ---
    if dy > 0:
        # Source smaller → pad
        top = dy // 2
        bottom = dy - top
        source = np.pad(source, ((top, bottom), (0, 0), (0, 0)),
                        mode='constant', constant_values=pad_value)
        pad_src[0] = [top, bottom]
    elif dy < 0:
        # Source larger → crop
        crop = abs(dy)
        top = crop // 2
        bottom = crop - top
        source = source[top:src_h - bottom, :, :]
        pad_src[0] = [-top, -bottom]

    # --- X-axis (width) ---
    if dx > 0:
        # Source smaller → pad
        left = dx // 2
        right = dx - left
        source = np.pad(source, ((0, 0), (left, right), (0, 0)),
                        mode='constant', constant_values=pad_value)
        pad_src[1] = [left, right]
    elif dx < 0:
        # Source larger → crop
        crop = abs(dx)
        left = crop // 2
        right = crop - left
        source = source[:, left:src_w - right, :]
        pad_src[1] = [-left, -right]

    # Now both have identical shape
    assert source.shape[:2] == target.shape[:2], "Source should now match target size."

    padding_params = {
        'pad_source': pad_src
    }

    return source, target, padding_params

def pad_images(
    image_1: np.ndarray,
    image_2: np.ndarray,
    pad_value: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, Dict, np.ndarray, np.ndarray]:
    """Pads two images to the same size and optionally adjusts landmarks accordingly."""
    y_size_1, x_size_1 = image_1.shape[:2]
    y_size_2, x_size_2 = image_2.shape[:2]
    
    pad_1, pad_2 = calculate_pad_value((y_size_1, x_size_1), (y_size_2, x_size_2))
    
    image_1 = np.pad(image_1, ((pad_1[0][0], pad_1[0][1]), (pad_1[1][0], pad_1[1][1]), (0, 0)), 
                     mode='constant', constant_values=pad_value)
    image_2 = np.pad(image_2, ((pad_2[0][0], pad_2[0][1]), (pad_2[1][0], pad_2[1][1]), (0, 0)), 
                     mode='constant', constant_values=pad_value)
    
  
    padding_params = {
        'pad_1': pad_1,
        'pad_2': pad_2
    }
    
    return image_1, image_2, padding_params

def pad_landmarks(
    padding_params,
    landmarks_1: np.ndarray = None,
    landmarks_2: np.ndarray = None,
    
) -> Tuple[np.ndarray, np.ndarray, Dict, np.ndarray, np.ndarray]:
   
    
    
    
    landmarks_1_padded = apply_padding_landmarks(landmarks_1, padding_params['pad_1']) if landmarks_1 is not None else None
    landmarks_2_padded = apply_padding_landmarks(landmarks_2, padding_params['pad_2']) if landmarks_2 is not None else None
    
   
    return  landmarks_1_padded, landmarks_2_padded

def remove_padding(image, pad_tuple):
    """Crop image or deformation field back to original (unpadded) size."""
    (top, bottom), (left, right) = pad_tuple
    h, w = image.shape[:2]
    cropped = image[top:h - bottom, left:w - right]
    return cropped
