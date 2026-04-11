Contributing
============

Contributions are welcome!  Please read these guidelines before opening a
pull request.

Getting started
---------------

1. Fork the repository on GitHub.
2. Create a feature branch::

      git checkout -b feature/my-new-feature

3. Make your changes, add tests, and ensure the test suite passes::

      pytest tests/

4. Open a pull request against the ``main`` branch.

Code style
----------

* Follow `PEP 8 <https://peps.python.org/pep-0008/>`_.
* Write docstrings in *NumPy* or *Google* style so that Sphinx autodoc can
  render them correctly.
* Keep functions small and focused.

Reporting issues
----------------

Use the GitHub issue tracker at
`github.com/eshasadia/CORE/issues <https://github.com/eshasadia/CORE/issues>`_.
Please include:

* A minimal reproducible example.
* Your Python and OS versions.
* The full traceback if applicable.

Citation
--------

If you use CORE in your research please cite:

.. code-block:: bibtex

   @misc{nasir2025corecelllevelcoarsetofine,
         title={CORE - A Cell-Level Coarse-to-Fine Image Registration Engine
                for Multi-stain Image Alignment},
         author={Esha Sadia Nasir and Behnaz Elhaminia and Mark Eastwood and
                 Catherine King and Owen Cain and Lorraine Harper and
                 Paul Moss and Dimitrios Chanouzas and David Snead and
                 Nasir Rajpoot and Adam Shephard and Shan E Ahmed Raza},
         year={2025},
         eprint={2511.03826},
         archivePrefix={arXiv},
         primaryClass={q-bio.QM},
         url={https://arxiv.org/abs/2511.03826},
   }

License
-------

CORE is released under the **MIT License**.  See ``LICENSE`` for the full
terms.
