# thyroid-classification

This repository contains the code for the paper titled "**Machine-learning-based diagnosis of thyroid fine-needle aspiration biopsy synergistically by Papanicolaou staining and refractive index distribution**", published in [*Scientific Reports*](https://doi.org/10.1038/s41598-023-36951-2).

For sample data used in the paper, please refer to [thyroid-classification-data](https://github.com/thyroiddata2023/thyroid-classification-data).

## Usage

### Patch-level prediction

```bash
python3 main.py {nbs_BF_5, nbs_MIP_5}
```

### Cluster-level prediction

- Please refer [ClusterPrediction.ipynb](ClusterPrediction.ipynb).

## Pretrained models

Pretrained model weights can be downloaded on following urls:

- [Patch-level models](https://drive.google.com/file/d/1yy-EfhEG7EHyR2kLNvlo7MbtgOIwygXy/view?usp=sharing) (Both color and RI models)

- [Cluster-level models](https://drive.google.com/file/d/1Xe6Ftv7GzzVgLq8bNPJWrRyfD37DosG3/view?usp=share_link) (XGBoost, Random Forest, Support Vector Machine, Multi-layer Perceptron)
