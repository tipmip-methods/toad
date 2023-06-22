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
The active working branch is `develop`, with the release branch `main` only to
be used for releases. Features are to be developed in [separate feature branches](https://nvie.com/posts/a-successful-git-branching-model/). 

---
June 2022 ∙ [Sina Loriani](mailto:sina.loriani@pik-potsdam.de)