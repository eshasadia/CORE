Quick Start
===========

The fastest way to get started is to run the end-to-end notebooks inside
the ``notebooks/`` folder.  The steps below show the minimal Python API.

.. figure:: _static/images/coarse_fine_registration.png
   :alt: Coarse-to-fine registration strategy
   :align: center
   :width: 100%

   **Figure 1.** CORE's two-stage registration strategy.  The coarse stage performs rigid and
   elastic alignment at low resolution; the fine stage refines cell-level correspondence using
   nuclei point-sets and shape-aware coherent point drift (CPD).

1. Edit ``config.py``
---------------------

Set at least the two WSI paths and the desired resolutions:

.. code-block:: python

   # config.py
   SOURCE_WSI_PATH = "/path/to/source.tiff"
   TARGET_WSI_PATH = "/path/to/target.tiff"

2. Load images
--------------

.. code-block:: python

   from core.preprocessing.preprocessing import load_wsi_images
   from core.config import PREPROCESSING_RESOLUTION

   source_wsi, target_wsi, source, target = load_wsi_images(
       SOURCE_WSI_PATH, TARGET_WSI_PATH, PREPROCESSING_RESOLUTION
   )

3. Pad images to a common canvas
---------------------------------

.. code-block:: python

   from core.preprocessing.padding import pad_images

   source_prep, target_prep, padding_params = pad_images(source, target)

4. Extract tissue masks
-----------------------

.. figure:: _static/images/tissue_mask_example.png
   :alt: Prompt-based tissue masking
   :align: center
   :width: 95%

   **Figure 2.** Prompt-based tissue segmentation.  VisionAgent uses a text prompt to isolate
   the tissue region, producing a binary mask that is then used to guide registration.

.. code-block:: python

   from core.preprocessing.preprocessing import extract_tissue_masks

   source_mask, target_mask = extract_tissue_masks(
       source_prep, target_prep, artefacts=False
   )

5. Coarse rigid registration
-----------------------------

.. code-block:: python

   from core.registration.registration import perform_rigid_registration

   moving_img_transformed, final_transform = perform_rigid_registration(
       source_prep, target_prep, source_mask, target_mask
   )

6. Coarse non-rigid (elastic) registration
-------------------------------------------

.. code-block:: python

   from core.registration.nonrigid import elastic_image_registration

   displacement_field, warped_source = elastic_image_registration(
       moving_img_transformed, target_prep
   )

7. Fine nuclei-level registration (optional)
---------------------------------------------

Fine registration refines alignment at the cell level using precomputed
nuclei coordinates and shape-aware point-set registration.  It requires
nuclei CSV files with ``global_x`` and ``global_y`` columns.

.. code-block:: python

   from core.preprocessing.nuclei_analysis import load_nuclei_coordinates
   from core.registration.registration import perform_shape_aware_registration
   from core.registration.nonrigid import compute_deformation_and_apply
   from core.preprocessing.padding import pad_landmarks
   import core.utils.util as util

   moving_df = load_nuclei_coordinates(MOVING_NUCLEI_CSV)
   fixed_df  = load_nuclei_coordinates(FIXED_NUCLEI_CSV)

   deformation_field, moving_updated, fixed_points, moving_points = \
       compute_deformation_and_apply(
           source_prep, final_transform, displacement_field,
           moving_df, fixed_df, padding_params, util, pad_landmarks
       )

   _, shape_transform, shape_coords = perform_shape_aware_registration(
       fixed_points, moving_updated, shape_weight=0.3,
       max_iterations=100, tolerance=1e-11
   )

8. Evaluate
-----------

.. figure:: _static/images/evaluation_tre.png
   :alt: Target Registration Error before and after registration
   :align: center
   :width: 95%

   **Figure 3.** Target Registration Error (TRE) before and after CORE registration.  Blue dots
   are fixed landmark positions; crosses show the corresponding moving landmarks.  A good
   registration brings the two sets of points into close agreement.

.. code-block:: python

   from core.evaluation.evaluation import evaluate_registration_tre

   metrics = evaluate_registration_tre(
       fixed_points, moving_points, final_transform,
       target_shape=target_prep.shape
   )
   print(f"TRE before: {metrics['tre_initial']:.4f}")
   print(f"TRE after : {metrics['tre_final']:.4f}")

Example notebooks
-----------------

Detailed walkthroughs are available in the ``notebooks/`` directory:

* ``notebooks/1-WSI_Registration.ipynb`` – coarse rigid and elastic registration demo
* ``notebooks/2-WSI_Registration.ipynb`` – nuclei-level fine registration demo
