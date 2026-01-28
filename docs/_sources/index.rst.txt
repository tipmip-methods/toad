.. TOAD documentation master file, created by
   Lukas Röhrich, November 2023

Home
====

**TOAD** (**T**\ ipping and **O**\ ther **A**\ brupt events **D**\ etector) is a Python framework for detecting and clustering spatio-temporal patterns in gridded Earth-system datasets.

.. note::
   For general information, project overview, and the latest updates, visit the `TOAD GitHub repository <https://github.com/tipmip-methods/toad>`_.

   If you're new to TOAD, start with the :doc:`installation` guide and then follow the :doc:`quick_start` tutorial.

What's in this documentation?
------------------------------

This documentation provides comprehensive guides for using TOAD in your research:

* :doc:`installation` - Installation instructions for different environments
* :doc:`quick_start` - Get started with TOAD in minutes
* :doc:`tutorials` - Detailed tutorials covering core concepts and advanced usage
* :doc:`api_ref` - Complete API reference for all classes and functions
* :doc:`scientific_ref` - Scientific references and methodology details
* :doc:`release_notes` - Version history and changelog

The TOAD Pipeline
------------------

.. image:: resources/TOAD_pipeline.png
   :alt: TOAD pipeline workflow
   :align: center
   :width: 100%

TOAD provides a structured workflow for analyzing Earth-system data:

1. **Shift Detection**: Identify abrupt transitions at individual grid cells using configurable detection methods
2. **Clustering**: Group detected shifts spatially and temporally to reveal cohesive patterns
3. **Aggregation & Synthesis**: Aggregate results across multiple datasets, models, or methods to produce consensus clusters


Getting Help
------------

* **Documentation**: Browse the sections above or use the search function
* **GitHub Issues**: Report bugs or request features on `GitHub <https://github.com/tipmip-methods/toad/issues>`_
* **Source Code**: View the `source code <https://github.com/tipmip-methods/toad>`_ on GitHub
