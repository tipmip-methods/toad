Installation Guide
==================

TOAD
----

TOAD is not yet published on PyPI or conda, but you can easily install it directly from GitHub. 

**Recommended installation (using HTTPS, works on Linux, macOS, and Windows):**

.. code:: bash

    git clone https://github.com/tipmip-methods/toad.git
    cd toad
    pip install .

**Developer mode installation** (if you want to modify TOAD or contribute):

.. code:: bash

    git clone https://github.com/tipmip-methods/toad.git
    cd toad
    pip install -e .[dev]

The ``-e`` flag installs the package in "editable" mode. Changes to the source code are reflected immediately.
