# TOAD
**T**ipping and **o**ther **a**brupt events **D**etector

## Installation
Full installation, including the githash-labeled outputs (recommended):
```bash
pip install -e 'git+ssh://git@gitlab.pik-potsdam.de/sinal/toad.git#egg=toad[vc_labels]'
```

Minimal installation:
```bash
pip install 'git+ssh://git@gitlab.pik-potsdam.de/sinal/toad.git'
```

## Version information
**Version 0.1 [Oct 2022]** New repository after major refactoring. Working abrupt shift detection based on `asdetect` with an evaluation pipeline that adds the detection time series as auxiliary variable to a dataset. The git hash is additionally saved as an attribute.

---
October 2022 ∙ [Sina Loriani](mailto:sina.loriani@pik-potsdam.de)