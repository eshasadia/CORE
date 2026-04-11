"""
Registration algorithms for WSI alignment
"""

import cv2
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from pycpd import DeformableRegistration
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from scipy.optimize import minimize
import core.registration.rigid as rigid
import pandas as pd


# ---------------------- Rigid Registration ---------------------- #

def perform_rigid_registration(source_prep, target_prep, source_mask, target_mask):
    """
    Perform rigid registration between source and target images.

    Args:
        source_prep (np.ndarray): Preprocessed source image.
        target_prep (np.ndarray): Preprocessed target image.
        source_mask (np.ndarray): Source tissue mask.
        target_mask (np.ndarray): Target tissue mask.

    Returns:
        tuple: (warped_image, transformation_matrix)
    """
    moving_img_transformed, final_transform, _ = rigid.rigid_registration(
        source_prep, target_prep, source_mask, target_mask
    )

    height, width = source_prep.shape[:2]
    warped = cv2.warpAffine(source_prep, final_transform[0:-1], (width, height))

    return warped, final_transform

# ---------------------- Shape-Aware Registration ----------------------
def perform_shape_aware_registration(fixed_df, moving_df, shape_attribute='area',
                                     shape_weight=0.3, max_iterations=50, tolerance=1e-6):
    registrator = rigid.ShapeAwarePointSetRegistration(
        fixed_df, moving_df,
        shape_attribute=shape_attribute,
        shape_weight=shape_weight,
        max_iterations=max_iterations,
        tolerance=tolerance
    )
    registered_points = registrator.register()
    transform_matrix = registrator.get_transformation_matrix()
    coords = registered_points[['registered_x', 'registered_y']].values
    return registrator, transform_matrix, coords





# ---------------------- Mutual Nearest Neighbors ---------------------- #

def find_mutual_nearest_neighbors(fixed_points, moving_points):
    """
    Find mutual nearest neighbors (MNN) between two point sets.
    """
    nn_fixed_to_moving = NearestNeighbors(n_neighbors=1).fit(moving_points)
    dist1, idx1 = nn_fixed_to_moving.kneighbors(fixed_points)

    nn_moving_to_fixed = NearestNeighbors(n_neighbors=1).fit(fixed_points)
    dist2, idx2 = nn_moving_to_fixed.kneighbors(moving_points)

    mnn_pairs = [(i, j[0]) for i, j in enumerate(idx1) if idx2[j[0]][0] == i]
    fixed_mnn = np.array([fixed_points[i] for i, _ in mnn_pairs])
    moving_mnn = np.array([moving_points[j] for _, j in mnn_pairs])

    print(f"Matched MNN pairs: {len(mnn_pairs)}")
    return fixed_mnn, moving_mnn, mnn_pairs


# ---------------------- CPD Non-Rigid Registration ---------------------- #

def perform_cpd_registration(moving_points, fixed_points, beta=0.5, alpha=0.01,
                             max_iterations=200, tolerance=1e-9):
    """
    Perform Coherent Point Drift (CPD) non-rigid registration.
    """
    reg = DeformableRegistration(
        X=moving_points,
        Y=fixed_points,
        beta=beta,
        alpha=alpha,
        max_iterations=max_iterations,
        tol=tolerance
    )
    reg.register()
    return reg.TY


# ---------------------- Displacement Field ---------------------- #

def create_displacement_field(source_points, transformed_points, image_shape,
                              method='linear', sigma=10.0, max_displacement=10.0):
    """
    Generate dense displacement field from sparse non-rigid registration results.
    """
    displacements = transformed_points - source_points
    height, width = image_shape[:2] if len(image_shape) == 3 else image_shape

    y_coords, x_coords = np.mgrid[0:height, 0:width]
    grid_points = np.vstack((x_coords.ravel(), y_coords.ravel())).T

    dx_grid = griddata(source_points, displacements[:, 0], grid_points, method=method, fill_value=0).reshape(height, width)
    dy_grid = griddata(source_points, displacements[:, 1], grid_points, method=method, fill_value=0).reshape(height, width)

    dx_field = gaussian_filter(dx_grid, sigma=sigma)
    dy_field = gaussian_filter(dy_grid, sigma=sigma)

    magnitude = np.sqrt(dx_field**2 + dy_field**2)
    scale = np.minimum(1.0, max_displacement / (magnitude + 1e-8))
    dx_field *= scale
    dy_field *= scale

    return np.stack((dx_field, dy_field), axis=-1)


# ---------------------- Utility ---------------------- #

def convert_4x4_to_3x3_transform(transform_4x4):
    """
    Convert 4x4 transformation matrix to 3x3 affine matrix.
    """
    transform_3x3 = np.eye(3)
    transform_3x3[0:2, 0:2] = transform_4x4[0:2, 0:2]
    transform_3x3[0:2, 2] = transform_4x4[0:2, 3]
    return transform_3x3

# ---------------------- ICP Registration ---------------------- #

def perform_icp_registration(moving_points, fixed_points, threshold=50000.0):
    """
    Perform ICP registration on 2D point sets.

    Args:
        moving_points (np.ndarray): Moving point set (N, 2).
        fixed_points (np.ndarray): Fixed point set (M, 2).
        threshold (float): Maximum correspondence distance.

    Returns:
        tuple: (transformation_matrix, transformed_points)
    """
    # Convert to Open3D point clouds (z=0 for 2D)
    pcd_moving = o3d.geometry.PointCloud()
    pcd_moving.points = o3d.utility.Vector3dVector(
        np.hstack([moving_points, np.zeros((len(moving_points), 1))])
    )

    pcd_fixed = o3d.geometry.PointCloud()
    pcd_fixed.points = o3d.utility.Vector3dVector(
        np.hstack([fixed_points, np.zeros((len(fixed_points), 1))])
    )

    # Run ICP registration
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_moving, pcd_fixed, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    print("ICP Transformation Matrix:")
    print(reg_p2p.transformation)

    # Apply transformation to moving points
    moving_points_hom = np.hstack([
        moving_points,
        np.zeros((len(moving_points), 1)),  # z=0
        np.ones((len(moving_points), 1))    # homogeneous coordinate
    ])

    transformed_points = (reg_p2p.transformation @ moving_points_hom.T).T[:, :2]

    return reg_p2p.transformation, transformed_points

