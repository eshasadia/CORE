"""
Nuclei detection and analysis functions (tissue-aware, watershed-based)
with accurate area estimation.
"""

import os
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from skimage.filters import threshold_local
from skimage.morphology import h_maxima, h_minima
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from tiatoolbox.tools import patchextraction
from tiatoolbox.tools.registration.wsi_registration import AffineWSITransformer
import core.utils.util as util
from core.config import FIXED_THRESHOLD, MOVING_THRESHOLD, MIN_NUCLEI_AREA, GAMMA_CORRECTION
from core.preprocessing.preprocessing import process_nuclei_patch


# -----------------------------
# Patch extraction
# -----------------------------
def extract_patches_from_wsi(wsi, mask, patch_size=(1000, 1000), stride=(1000, 1000)):
    patch_extractor = patchextraction.get_patch_extractor(
        input_img=wsi,
        method_name="slidingwindow",
        patch_size=patch_size,
        input_mask=mask,
        stride=stride,
        resolution=0
    )
    return patch_extractor


# -----------------------------
# Watershed-based nuclei detection with area
# -----------------------------
def detect_nuclei_patch_watershed(img, min_area=25):
    """
    Detect nuclei in a patch using watershed and estimate area via contours.

    Returns:
        nuclei_stats: list of dicts {'area': area}
        nuclei_centroids: np.array of centroids [[x1, y1], [x2, y2], ...]
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_smooth = cv2.GaussianBlur(gray, (3,3), 0)

    # Adaptive threshold
    block_size = 51
    adaptive_thresh = threshold_local(gray_smooth, block_size, offset=5)
    binary = gray < adaptive_thresh

    # Distance transform and watershed
    distance = ndi.distance_transform_edt(binary)
    markers, _ = ndi.label(binary)
    labels = watershed(-distance, markers, mask=binary)

    nuclei_stats = []
    nuclei_centroids = []

    for label in np.unique(labels):
        if label == 0:
            continue  # skip background
        mask = np.uint8(labels == label)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        nuclei_centroids.append([cx, cy])
        nuclei_stats.append({'area': area})

    return nuclei_stats, np.array(nuclei_centroids)


# -----------------------------
# Patch processing
# -----------------------------
def process_fixed_patch(patch_extractor, patch_idx):
    fixed_img = patch_extractor[patch_idx]
    coords = patch_extractor.coordinate_list[patch_idx]
    
    fixed_stats, fixed_centroids = detect_nuclei_patch_watershed(fixed_img, min_area=MIN_NUCLEI_AREA)
    
    nuclei_data = []
    for i, centroid in enumerate(fixed_centroids):
        x, y = centroid
        global_x = coords[0] + x
        global_y = coords[1] + y
        nuclei_data.append({
            'patch_id': patch_idx,
            'patch_x1': coords[0],
            'patch_y1': coords[1],
            'patch_x2': coords[2],
            'patch_y2': coords[3],
            'local_x': x,
            'local_y': y,
            'global_x': global_x,
            'global_y': global_y,
            'area': fixed_stats[i]['area']
        })
    return nuclei_data


def process_moving_patch(patch_extractor, tfm, patch_idx):
    coords = patch_extractor.coordinate_list[patch_idx]
    location = (coords[0], coords[1])
    size = (coords[2] - coords[0], coords[3] - coords[1])
    
    moving_img = tfm.read_rect(location, size, resolution=40, units="power")
    moving_stats, moving_centroids = detect_nuclei_patch_watershed(moving_img, min_area=MIN_NUCLEI_AREA)
    
    nuclei_data = []
    for i, centroid in enumerate(moving_centroids):
        x, y = centroid
        global_x = coords[0] + x
        global_y = coords[1] + y
        nuclei_data.append({
            'patch_id': patch_idx,
            'patch_x1': coords[0],
            'patch_y1': coords[1],
            'patch_x2': coords[2],
            'patch_y2': coords[3],
            'local_x': x,
            'local_y': y,
            'global_x': global_x,
            'global_y': global_y,
            'area': moving_stats[i]['area']
        })
    return nuclei_data


def _process_patch_pair(args):
    """Process a single (fixed, moving) patch pair; used by the thread pool."""
    fixed_patch_extractor, tfm, patch_idx = args
    fixed_nuclei, moving_nuclei = [], []
    try:
        fixed_nuclei = process_fixed_patch(fixed_patch_extractor, patch_idx)
        moving_nuclei = process_moving_patch(fixed_patch_extractor, tfm, patch_idx)
    except Exception as e:
        print(f"Error processing patch {patch_idx}: {e}")
    return fixed_nuclei, moving_nuclei


def process_nuclei_in_patches(fixed_patch_extractor, tfm, start_index=0, end_index=None):
    if end_index is None:
        end_index = len(fixed_patch_extractor) - 1

    patch_indices = list(range(start_index, end_index + 1))
    args_list = [(fixed_patch_extractor, tfm, idx) for idx in patch_indices]

    all_fixed_nuclei_data = []
    all_moving_nuclei_data = []

    # Use ThreadPoolExecutor: I/O-bound WSI reads benefit from threads without pickling overhead
    max_workers = min(os.cpu_count() or 1, len(patch_indices))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for fixed_nuclei, moving_nuclei in executor.map(_process_patch_pair, args_list):
            all_fixed_nuclei_data.extend(fixed_nuclei)
            all_moving_nuclei_data.extend(moving_nuclei)

    return all_fixed_nuclei_data, all_moving_nuclei_data


# -----------------------------
# CSV utilities
# -----------------------------
def save_nuclei_data_to_csv(fixed_nuclei_data, moving_nuclei_data, 
                            fixed_csv_path, moving_csv_path):
    fixed_nuclei_df = pd.DataFrame(fixed_nuclei_data)
    moving_nuclei_df = pd.DataFrame(moving_nuclei_data)
    
    fixed_nuclei_df.to_csv(fixed_csv_path, index=False)
    moving_nuclei_df.to_csv(moving_csv_path, index=False)
    
    print(f"Saved {len(fixed_nuclei_data)} fixed nuclei to {fixed_csv_path}")
    print(f"Saved {len(moving_nuclei_data)} moving nuclei to {moving_csv_path}")


def load_nuclei_coordinates(csv_path):
    df = pd.read_csv(csv_path)
    if 'area' not in df.columns:
        df['area'] = 1.0
    return df


def extract_nuclei_points(df, columns=['global_x', 'global_y']):
    return df[columns].to_numpy()


def subsample_nuclei(df, n_samples, random_state=42):
    if len(df) <= n_samples:
        return df
    return df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)


def create_nuclei_dataframe_from_points(points, area_values=None):
    df = pd.DataFrame(points, columns=['global_x', 'global_y'])
    if area_values is not None:
        df['area'] = area_values
    else:
        df['area'] = 1.0
    return df
