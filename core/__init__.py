"""
CORE – Coarse-to-fine Registration of whole-slide images.

Public API
----------
The most commonly used functions are re-exported here for convenience::

    from core import (
        load_wsi_images,
        extract_tissue_masks,
        perform_rigid_registration,
        elastic_image_registration,
        evaluate_registration_tre,
    )

Imports are deferred so that sub-packages with heavy optional dependencies
(tiatoolbox, torch, vision_agent …) are only loaded when actually used.
"""

__all__ = [
    "load_wsi_images",
    "extract_tissue_masks",
    "perform_rigid_registration",
    "elastic_image_registration",
    "evaluate_registration_tre",
]


def __getattr__(name):
    """Lazy-import public API symbols on first access."""
    _registry = {
        "load_wsi_images": ("core.preprocessing.preprocessing", "load_wsi_images"),
        "extract_tissue_masks": ("core.preprocessing.preprocessing", "extract_tissue_masks"),
        "perform_rigid_registration": ("core.registration.registration", "perform_rigid_registration"),
        "elastic_image_registration": ("core.registration.nonrigid", "elastic_image_registration"),
        "evaluate_registration_tre": ("core.evaluation.evaluation", "evaluate_registration_tre"),
    }
    if name in _registry:
        module_path, attr = _registry[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'core' has no attribute {name!r}")
