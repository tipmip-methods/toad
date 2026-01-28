Quick Start
===========

This guide will get you up and running with TOAD in just a few minutes. For more detailed examples and explanations, see the :doc:`tutorials` section.

Minimal Example
---------------

Here's a simple example that demonstrates the core TOAD workflow:

.. code-block:: python

    from toad import TOAD
    from toad.shifts import ASDETECT
    from sklearn.cluster import HDBSCAN

    # Initialize TOAD object with your data file
    td = TOAD("data.nc")

    # Detect abrupt shifts using ASDETECT method
    td.compute_shifts("tas", method=ASDETECT())

    # Cluster detected shifts using HDBSCAN
    td.compute_clusters(
        var="tas",
        method=HDBSCAN(min_cluster_size=10),
    )

    # Visualize the results
    td.plot.overview("tas")

What This Does
--------------

1. **``TOAD("data.nc")``**: Loads your gridded data from a NetCDF file
2. **``compute_shifts()``**: Analyzes each grid cell to detect abrupt transitions
3. **``compute_clusters()``**: Groups spatially and temporally co-occurring shifts
4. **``plot.overview()``**: Creates a visualization of the detected clusters

Next Steps
----------

- **Learn the basics**: Check out the :doc:`tutorials/basics` tutorial for a comprehensive introduction
- **Explore visualization**: See the plotting capabilities in the tutorials
- **Customize methods**: Learn how to implement custom clustering or shift detection methods
- **API reference**: Browse the :doc:`api_ref` for detailed documentation of all classes and methods

Data Format
-----------

TOAD expects input data as:

- **NetCDF files** (``.nc``) readable by xarray
- **xarray Dataset or DataArray** objects
- Data structured as 3D arrays: ``space × space × time``

The ``time`` dimension can represent actual time or any other forcing variable or bifurcation parameter.

For more information about data requirements and formats, see the :doc:`tutorials/basics` tutorial.
