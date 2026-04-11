Configuration Reference
=======================

All tuneable parameters live in ``core/config.py``.  Edit that file before
running a registration pipeline.

File paths
----------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - Description
   * - ``SOURCE_WSI_PATH``
     - Absolute path to the *moving* (source) whole-slide image (``.tiff``).
   * - ``TARGET_WSI_PATH``
     - Absolute path to the *fixed* (target) whole-slide image (``.tiff``).
   * - ``FIXED_NUCLEI_CSV``
     - Path to a pre-computed nuclei CSV for the fixed image (optional).
   * - ``MOVING_NUCLEI_CSV``
     - Path to a pre-computed nuclei CSV for the moving image (optional).

Each CSV must contain at least the columns ``global_x`` and ``global_y``
(nucleus centroid coordinates in pixels at ``REGISTRATION_RESOLUTION``).
An optional ``area`` column enables shape-aware registration.

Resolution parameters
---------------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Parameter
     - Description
   * - ``PREPROCESSING_RESOLUTION``
     - MPP (microns-per-pixel) used for coarse alignment. Default: ``0.625``.
   * - ``REGISTRATION_RESOLUTION``
     - MPP used for nuclei detection and fine alignment. Default: ``40``.

``REGISTRATION_RESOLUTION`` must be strictly greater than
``PREPROCESSING_RESOLUTION``.

Patch parameters
----------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Parameter
     - Description
   * - ``PATCH_SIZE``
     - ``(height, width)`` of each patch in pixels. Default: ``(1000, 1000)``.
   * - ``PATCH_STRIDE``
     - Stride between patches. Default: ``(1000, 1000)``.
   * - ``VISUALIZATION_SIZE``
     - Canvas size for visualisation output. Default: ``(5000, 5000)``.

Nuclei detection parameters
-----------------------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Parameter
     - Description
   * - ``FIXED_THRESHOLD``
     - Binarisation threshold for the fixed image (0–255). Default: ``100``.
   * - ``MOVING_THRESHOLD``
     - Binarisation threshold for the moving image (0–255). Default: ``50``.
   * - ``MIN_NUCLEI_AREA``
     - Minimum nucleus area in pixels². Default: ``200``.
   * - ``GAMMA_CORRECTION``
     - Gamma value applied before nuclei detection. Default: ``0.4``.

``RegistrationParams`` class
-----------------------------

.. autoclass:: core.config.RegistrationParams
   :members:
   :undoc-members:

``VisualizationParams`` class
------------------------------

.. autoclass:: core.config.VisualizationParams
   :members:
   :undoc-members:
