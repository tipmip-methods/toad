# TOAD
**T**ipping and **o**ther **a**brupt events **D**etector. Tutorial how to use [here.](https://gitlab.pik-potsdam.de/sinal/toadtorial)

## Installation
Installation:
```bash
pip install 'git+ssh://git@gitlab.pik-potsdam.de/sinal/toad.git'
```

## Version information

**Version 0.2 [Jun 2023]** Working clustering based on `DBSCAN` with an
evaluation pipeline that adds the cluster labels as auxiliary variable to a
dataset. Also testing post-clustering evaluation methods (API might change!).

**Version 0.1 [Oct 2022]** New repository after major refactoring. Working
abrupt shift detection based on `asdetect` with an evaluation pipeline that adds
the detection time series as auxiliary variable to a dataset. The git hash is
additionally saved as an attribute.

## Repository information
We use [trunk-based
development](https://medium.com/@vafrcor2009/gitflow-vs-trunk-based-development-3beff578030b)
for our git workflow. This means we all work on the same branch (main), the
trunk, and push our code changes to it often. This way, we can keep our code up
to date. We also avoid having too many branches that can get messy and hard to
merge. We only create short-lived branches for small features or bug fixes, and
we merge them back to main as soon as they are done. To this end, each developer
issues pull-requests that are approved or rejected by the maintainer. Special
versions of the code can be then dedicated releases with version tags, allowing
others to use very specific versions of the code if needed.

---
June 2022 ∙ [Sina Loriani](mailto:sina.loriani@pik-potsdam.de)