# TOAD
**T**ipping and **O**ther **A**brupt events **D**etector. 

## Installation
Installation:
```bash
pip install (to be added)
```

Please see tutorial in [examples/basics.ipynb](examples/basics.ipynb).

### Simple use case
``` python
from toad import TOAD
from toad.shifts_detection.methods import ASDETECT
from toad.clustering.methods import HDBSCAN


td = TOAD("data.nc")
td.compute_shifts("temp", method=ASDETECT())
td.compute_clusters(
    var="temp",                                  
    method=HDBSCAN(min_cluster_size=25),        
    shifts_filter_func=lambda x: np.abs(x)>0.8,
)
```



## Development
```bash
git clone (to be added)
cd toad
pip install -e .
```
The `-e` flag installs the package in "editable" mode, which means changes to the source code are immediately reflected without needing to reinstall.


## Version information

**Version 0.3 [Nov 2024]** Major refactoring of the code to increase
ease of use and extendability. First version on Github. The main use-case now happens through
the `TOAD` object, which wraps the xarray dataset and provides analysis
functions etc. Added `HDBSCAN` clustering and basic plotting functionality
(still work in progress). Various classes/functions are still to be implemented,
but basic functionality of 0.2 is there, and few breaking changes are expected.

**Version 0.2 [Jun 2023]** Working clustering based on `DBSCAN` with an
evaluation pipeline that adds the cluster labels as auxiliary variable to a
dataset. Also testing post-clustering evaluation methods (API might change!).

**Version 0.1 [Oct 2022]** New repository after major refactoring. Working
abrupt shift detection based on `asdetect` with an evaluation pipeline that adds
the detection time series as auxiliary variable to a dataset. The git hash is
additionally saved as an attribute.

## Repository information
We use [trunk-based development](https://medium.com/@vafrcor2009/gitflow-vs-trunk-based-development-3beff578030b)
for our git workflow. This means we all work on the same branch (main), the
trunk, and push our code changes to it often. This way, we can keep our code up
to date. We also avoid having too many branches that can get messy and hard to
merge. We only create short-lived branches for small features or bug fixes, and
we merge them back to main as soon as they are done. To this end, each developer
issues pull-requests that are approved or rejected by the maintainer. Special
versions of the code can be then dedicated releases with version tags, allowing
others to use very specific versions of the code if needed.


## Note on using older version of TOAD
Version 0.2 of TOAD is still available on the `toad-alpha` branch. The package there has been renamed from `toad` to `toad-alpha` such that it can be used alongside the most recent version of toad if cloned locally: 
```bash
git clone (to be added) :: toad-alpha
cd toad
pip install -e .
```
Then
``` python
import toad-alpha as toad
```

---
Nov 2024 ∙ [Jakob Harteg](mailto:jakob.harteg@pik-potsdam.de)

June 2022 ∙ [Sina Loriani](mailto:sina.loriani@pik-potsdam.de)
