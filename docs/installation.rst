Installation
============

Requirements
------------

* Python 3.10 or newer
* `Conda <https://docs.conda.io>`_ (recommended for environment management)
* A CUDA-capable GPU is strongly recommended for fast inference

Clone the repository
--------------------

.. code-block:: bash

   git clone https://github.com/eshasadia/CORE.git
   cd CORE

Create the Conda environment
----------------------------

The ``environment.yml`` file contains all required dependencies:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate core

Set API keys
------------

CORE's prompt-based tissue mask generation relies on the *VisionAgent* service.
Export your API key before running the pipeline:

.. code-block:: bash

   export VISION_AGENT_API_KEY="your-api-key"

Pre-trained UNet weights
------------------------

UNet-based tissue mask extraction weights are hosted on Hugging Face:
`eshasadianasir/CORE <https://huggingface.co/eshasadianasir/CORE/tree/main>`_.
Download the checkpoint and point ``UNET_WEIGHTS_PATH`` in ``config.py`` to the
downloaded file.
