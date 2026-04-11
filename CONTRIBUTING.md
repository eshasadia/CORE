# Contributing to CORE

Thank you for your interest in contributing to **CORE** (Coarse-to-fine Registration of whole-slide images)!  
This document covers how to set up your environment, coding standards, how to add tests, and how to propose a new registration method.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Project Layout](#project-layout)
3. [Coding Standards](#coding-standards)
4. [Running Tests](#running-tests)
5. [Adding a New Registration Method](#adding-a-new-registration-method)
6. [Pull Request Process](#pull-request-process)

---

## Getting Started

```bash
# 1. Fork the repository and clone your fork
git clone https://github.com/<your-username>/WSI_mIF_Reg.git
cd WSI_mIF_Reg

# 2. Create and activate the conda environment
conda env create -f environment.yml
conda activate core

# 3. Set the VisionAgent API key (needed for tissue-mask extraction)
export VISION_AGENT_API_KEY="<your-key>"

# 4. Install the package in editable mode (optional but recommended)
pip install -e .
```

---

## Project Layout

```
WSI_mIF_Reg/
├── core/
│   ├── config.py              # Global configuration and parameter classes
│   ├── preprocessing/
│   │   ├── preprocessing.py   # WSI loading, padding, stain normalisation
│   │   ├── tissuemask.py      # Tissue mask extraction (Florence-2, UNet, Otsu)
│   │   ├── stainnorm.py       # Macenko stain normaliser
│   │   ├── nuclei_analysis.py # Nuclei detection and patch processing
│   │   └── padding.py         # Image and landmark padding helpers
│   ├── registration/
│   │   ├── registration.py    # Rigid, ICP, CPD, shape-aware registration
│   │   ├── rigid.py           # Trimorph + XFeat coarse rigid registration
│   │   ├── nonrigid.py        # Elastic / non-rigid registration (VoxelMorph-style)
│   │   └── cpd.py             # Coherent Point Drift implementation
│   ├── evaluation/
│   │   └── evaluation.py      # TRE, rTRE, NGF metrics
│   ├── utils/
│   │   ├── util.py            # General utilities (deformation field ops, maths)
│   │   └── mha_wsi.py         # MHA deformation field ↔ WSI application
│   └── visualization/
│       └── visualization.py   # Matplotlib and Bokeh plotting helpers
├── tests/                     # pytest unit tests
├── notebooks/                 # Jupyter walkthrough notebooks
├── example.py                 # End-to-end usage script
└── apply_deformation_wsi.py   # CLI: apply MHA field to full-res WSI
```

---

## Coding Standards

* **Python ≥ 3.10** is required.
* Follow [PEP 8](https://peps.python.org/pep-0008/) style; maximum line length is **100** characters.
* Use **type annotations** for all new public functions and methods.
* Use the `logging` module rather than `print()` for diagnostic output:

  ```python
  import logging
  logger = logging.getLogger(__name__)
  logger.info("Processing %d patches", n_patches)
  ```

* Prefer **numpy vectorised operations** over Python loops when working on large arrays.
* Do not commit large binary data (model weights, full-res WSIs) to the repository.

---

## Running Tests

Tests live in the `tests/` directory and use `pytest`.

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_evaluation.py -v

# Run with coverage report
pytest tests/ --cov=core --cov-report=term-missing
```

Tests are also run automatically on every pull request via the CI workflow (`.github/workflows/tests.yml`).

---

## Adding a New Registration Method

1. **Create a new function or class** in the appropriate module under `core/registration/`.  
   If the method is substantial, create a dedicated file (e.g., `core/registration/my_method.py`).

2. **Follow the expected interface** — registration functions typically accept:
   - `fixed_points` / `moving_points` (numpy `ndarray`, shape `(N, 2)`) for point-based methods, or
   - `source_image` / `target_image` (numpy `uint8 RGB`, shape `(H, W, 3)`) for image-based methods.

3. **Export it** from `core/registration/registration.py` (or add it to `core/__init__.py` if it is a primary public API).

4. **Write unit tests** in `tests/test_registration.py` covering:
   - Correctness on a synthetic case (e.g., a known translation).
   - Edge cases (empty point sets, single points, etc.).

5. **Document it** in the docstring using the existing format (NumPy-style), and add a short usage example to `example.py` or a new notebook.

---

## Pull Request Process

1. Branch off `main` with a descriptive name, e.g. `feature/my-method` or `fix/bgr-rgb-conversion`.
2. Keep each PR focused on a single concern.
3. Ensure all tests pass locally (`pytest tests/`).
4. Ensure your changes do not introduce new linting errors.
5. Update the relevant documentation (docstrings, `README.md`, `CONTRIBUTING.md` if needed).
6. Open a pull request against `main` and fill in the PR template.

Thank you for helping make CORE better!
