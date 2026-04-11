"""
Configuration file for CORE
"""
import os

# Set the API key for Vision Agent only if not already present in the environment
if "VISION_AGENT_API_KEY" not in os.environ:
    os.environ["VISION_AGENT_API_KEY"] = ""

# File paths
SOURCE_WSI_PATH = ''
TARGET_WSI_PATH = ''

# Output Nuclei or Precomputed Nuclei CSV paths
# ─────────────────────────────────────────────
# Both CSV files must contain at least two columns:
#   ``global_x`` (float) – x coordinate of the nucleus centroid in pixels at
#                           the full registration resolution (REGISTRATION_RESOLUTION).
#   ``global_y`` (float) – y coordinate of the nucleus centroid in pixels.
# An optional ``area`` (float) column is used for shape-aware registration and
# for colour-mapping in the visualisation plots.  When absent it defaults to 1.0.
FIXED_NUCLEI_CSV = ''
MOVING_NUCLEI_CSV = ''

# Registration parameters
#  initial resolution for coarse registration
PREPROCESSING_RESOLUTION = 0.625
#  High resolution for nuclei estimation and shape-aware registration
REGISTRATION_RESOLUTION = 40

assert PREPROCESSING_RESOLUTION > 0, "PREPROCESSING_RESOLUTION must be positive."
assert REGISTRATION_RESOLUTION > 0, "REGISTRATION_RESOLUTION must be positive."
assert REGISTRATION_RESOLUTION > PREPROCESSING_RESOLUTION, (
    "REGISTRATION_RESOLUTION must be greater than PREPROCESSING_RESOLUTION."
)

PATCH_SIZE = (1000, 1000)
PATCH_STRIDE = (1000, 1000)
VISUALIZATION_SIZE = (5000, 5000)

# Nuclei detection parameters
#  needs to changed wrt to the datasets
FIXED_THRESHOLD = 100
MOVING_THRESHOLD = 50
MIN_NUCLEI_AREA = 200
GAMMA_CORRECTION = 0.4

assert 0 < FIXED_THRESHOLD < 256, "FIXED_THRESHOLD must be in range (0, 256)."
assert 0 < MOVING_THRESHOLD < 256, "MOVING_THRESHOLD must be in range (0, 256)."
assert MIN_NUCLEI_AREA >= 0, "MIN_NUCLEI_AREA must be non-negative."


# Registration algorithm parameters
class RegistrationParams:
    # MNN sampling – maximum number of nuclei sampled per image for the mutual
    # nearest-neighbour matching step.  Reducing this speeds up large datasets.
    MNN_SAMPLE_SIZE = 5000

    # Displacement field parameters
    DISPLACEMENT_SIGMA = 10.0
    MAX_DISPLACEMENT = 500.0
    INTERPOLATION_METHOD = 'linear'


# Visualization parameters
class VisualizationParams:
    FIGURE_WIDTH = 900
    FIGURE_HEIGHT = 700
    POINT_SIZE_SMALL = 2
    POINT_SIZE_MEDIUM = 3
    POINT_SIZE_LARGE = 10
    ALPHA = 0.6

    # Colors
    FIXED_COLOR = "blue"
    MOVING_COLOR = "red"
    RIGID_COLOR = "green"
    NONRIGID_COLOR = "orange"
