# <img src="https://github.com/user-attachments/assets/2f7f299f-1f04-4a7d-b650-54b8ce96c570" width="30" height="30"> CORE - A Cell-Level Coarse-to-Fine Image Registration Engine for Multi-stain Image Alignment
[![arXiv](https://img.shields.io/badge/arXiv-2403.05780-b31b1b.svg)](https://arxiv.org/html/2511.03826v2)
[![Greetings](https://github.com/eshasadia/CORE/actions/workflows/greetings.yml/badge.svg)](https://github.com/eshasadia/CORE/actions/workflows/greetings.yml)
![License](https://img.shields.io/badge/License-MIT-green.svg)
[![pages-build-deployment](https://github.com/eshasadia/CORE/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/eshasadia/CORE/actions/workflows/pages/pages-build-deployment)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://eshasadia.github.io/CORE/)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
[![Last Commit](https://img.shields.io/github/last-commit/eshasadia/CORE/main.svg)](https://github.com/eshasadia/CORE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Conda](https://img.shields.io/badge/conda-environment-yellowgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![CUDA](https://img.shields.io/badge/CUDA-supported-blue)
[![Florence-SAM](https://img.shields.io/badge/Florence--SAM-Project-blue)](https://github.com/landing-ai/vision-agent)
![Build](https://img.shields.io/badge/build-passing-brightgreen)






## News
📢 November 2025 — CORE Released as Open-Source
The first public release of CORE, a unified coarse-to-fine multi-stain image registration engine, is now available. This release includes prompt-guided mask generation, accelerated features based coarse alignment, nuclei-level refinement, and real-time deformation visualization.

📝 November 2025 — Updated Preprint Available on arXiv. The team has released an updated version of the CORE preprint, expanding on the architecture, benchmarks, and qualitative results. Check out the newest version here: [arXiv:2403.05780](https://arxiv.org/html/2511.03826v2).

🎥 New [TIAViz Integration Demo](https://tiademos.dcs.warwick.ac.uk/bokeh_app?demo=WSIReg) - Added a full registration workflow demo showing real-time deformation fields and alignment quality inside TIAViz, enabling seamless analysis for whole-slide images.

🧪 Sample Notebooks Added - End-to-end Jupyter notebooks for coarse and fine alignment have been added, making it easier for users to experiment with CORE immediately.

## Introduction

CORE is a fast and accurate coarse-to-fine image registration engine designed for aligning multi-stain whole-slide images. It combines prompt-based tissue masking, rapid coarse alignment, and nuclei-level fine registration to deliver precise cell-level correspondence across stains. With real-time deformation visualization and easy integration, CORE enables reliable multi-stain analysis for digital pathology workflows.

## Features

- **Prompt-based Tissue Mask Extraction.** 
- **Fast coarse level multi-stain image registration.** 
- **Fine-grained Nuclei-level precise alignment on re-stained sections and tissue alignment on consecutive sections.**
- **Real time deformation estimation and Registration visualisation.**

![CORE Architecture](https://github.com/user-attachments/assets/ffeca69d-e1b2-4260-bbd2-edf3fa28f76a)

## CORE VISUALIZATION
<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/2df6cf94-c855-46ca-9b5a-4782c9f69ff3" alt="Registration Visualization on TIAViz">
</div>


## Installation

1. Clone the repo.
2. change directory to project directory
3. Create conda enivornment for installing the required dependencies using the following command
   ```
   conda env create -f environment.yml
   conda activate core
   ```

### Set API Keys as Environment Variables
1. For our prompt-based tissue mask generation. You must set the VisionAgent API key as environment variables. Each operating system offers different ways to do this.
Here is the code for setting the variables:

```bash
export VISION_AGENT_API_KEY="your-api-key"
```
2. For UNet based tissue mask extraction we have made the weights publicly available on hugging face. [CORE](https://huggingface.co/eshasadianasir/CORE/tree/main)

## Configuration
Edit `config.py` to set your file paths and  resolution parameters:
```python
# Update these paths to match your data
SOURCE_WSI_PATH = "/path/to/your/source_wsi.tiff"
TARGET_WSI_PATH = "/path/to/your/target_wsi.tiff"
```

## Usage
Example of both coarse and fine registration have been placed under the notebooks folder.

### Web tool (on-the-fly WSI registration)
You can run a browser-based single-pair registration tool powered by Bokeh:

```bash
bokeh serve --show web_tool.py
```

Then provide:
- source WSI path (moving image)
- target WSI path (fixed image)
- output directory and registration parameters

The tool runs CORE registration and writes:
- combined deformation field (`.mha`)
- registered whole-slide image (`.ome.tiff`)
- optional checkerboard preview image

## How to Cite
```bibtex
@misc{nasir2025corecelllevelcoarsetofine,
      title={CORE - A Cell-Level Coarse-to-Fine Image Registration Engine for Multi-stain Image Alignment}, 
      author={Esha Sadia Nasir and Behnaz Elhaminia and Mark Eastwood and Catherine King and Owen Cain and Lorraine Harper and Paul Moss and Dimitrios Chanouzas and David Snead and Nasir Rajpoot and Adam Shephard and Shan E Ahmed Raza},
      year={2025},
      eprint={2511.03826},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM},
      url={https://arxiv.org/abs/2511.03826}, 
}
```


## CORE Registration DEMO
<video src='https://github.com/user-attachments/assets/140e3c40-40e1-429f-a49b-1fd9ede790ff' width=180/>
