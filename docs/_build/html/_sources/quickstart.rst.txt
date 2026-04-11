Quick Start
===========

The fastest way to get started is to run the end-to-end notebooks inside
the ``notebooks/`` folder.  The steps below show the minimal Python API.

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

   from core import load_wsi_images

   fixed_img, moving_img = load_wsi_images()

3. Extract tissue masks
-----------------------

.. code-block:: python

   from core import extract_tissue_masks

   fixed_mask, moving_mask = extract_tissue_masks(fixed_img, moving_img)

4. Coarse (rigid) registration
-------------------------------

.. code-block:: python

   from core import perform_rigid_registration

   rigid_result = perform_rigid_registration(fixed_img, moving_img,
                                             fixed_mask, moving_mask)

5. Fine (non-rigid) registration
---------------------------------

.. code-block:: python

   from core import elastic_image_registration

   nonrigid_result = elastic_image_registration(rigid_result)

6. Evaluate
-----------

.. code-block:: python

   from core import evaluate_registration_tre

   tre = evaluate_registration_tre(nonrigid_result)
   print(f"Target Registration Error: {tre:.4f}")

Example notebooks
-----------------

Detailed walkthroughs are available in the ``notebooks/`` directory:

* ``notebooks/coarse_registration.ipynb`` – coarse alignment demo
* ``notebooks/fine_registration.ipynb`` – nuclei-level fine alignment demo
