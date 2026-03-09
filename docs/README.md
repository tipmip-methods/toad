

Run locally using command
$ cd docs
$ sphinx-autobuild source build/html --open-browser

To get tutorials to show, you will need to copy them over manually using
$ make copy-tutorials

**If the sidebar or main navigation doesn't update** (e.g. after adding or renaming pages): do a clean rebuild with ``make clean && make html``, or run ``sphinx-build -E source build/html`` to ignore the cached environment. Then restart sphinx-autobuild. A hard browser refresh (Cmd+Shift+R) may also help.