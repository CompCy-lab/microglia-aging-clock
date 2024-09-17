# Translating Single-Cell Microglia Profiles into Aging Clocks
<p>
  <img src="https://github.com/CompCy-lab/microglia-aging-clock/blob/main/microglia_img.png?raw=True" width="350" />
</p>

## Overview

We explore how various unsupervised featurization strategies can be used to translate per-sample single-cell microglia profiles into models of age. Here, we provide examples for generating features from microglia from scLKME, frequency, pseudobulk++ (our new proposed method), and classical pseudobulk.

## Installation
Clone the repository:

```
git clone https://github.com/CompCy-lab/microglia-aging-clock
```

Once you have cloned the repository, change your working directory as,

```
cd microglia-aging-clock
```

Install the required dependencies:

```
pip install -r requirements.txt
```

## Datasets
All three datasets from our original study can be found in a [Zenodo repository](https://zenodo.org/records/12811383?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjAxMGRkODUzLTY0NjAtNGU5MC1hNzgzLWUwMzhmZTZlOTZlYiIsImRhdGEiOnt9LCJyYW5kb20iOiI1OWE5ZDVmYmVjMmIzNjUxNzUxNTllZGMzNGYyMjgwNSJ9.rB0clZfO5rsht2svUh6rI5oQxgTUOrtHrzAME0Ms0PE9wk5_gl7lU5z0h9TCpMBsKs_5__psMC4LQp7kYJySCQ). The examples below use the [Hammond dataset, Immunity 2019](https://www.sciencedirect.com/science/article/pii/S1074761318304850?via%3Dihub). 

## Example for frequency features

Read in .h5ad file `Hammond_adata_merge_with_donors.h5ad` containing single-cell data in annData format. We assumed cells have clusters with key `leiden`, sample id indicating the sample they came from stored with key `sample_id`, donor indicating the donor they came from stored with key `donor`, and associated ages of the sample they came from stored with key `age` for categorical age and with key `age_continuous` for continuous age.

```python
import anndata
import os
adata_merge = anndata.read_h5ad(os.path.join('data','Hammond_adata_merge_with_donors.h5ad'))
```

Alternatively:

```python
import scanpy as sc
import os
adata_merge = sc.read_h5ad(os.path.join('data','Hammond_adata_merge_with_donors.h5ad'))
```

Given the keys described above with metadata about the cells, prepare a pandas data frame to input to the frequency function:

```python
from microglia_aging_clock import *
data = pd.DataFrame(adata_merge.X, columns=adata_merge.var_names, index=adata_merge.obs_names)
data['sample_id'] = adata_merge.obs['sample_id'].copy()
data['cluster_label'] = adata_merge.obs['leiden'].copy()
data['age'] = adata_merge.obs['age'].copy()

if 'donor' in adata_merge.obs.columns:
    data['donor'] = adata_merge.obs['donor'].copy()
```

Extract frequency features and an associated list of ages:

```python
frequencyFeatures = frequency_feats(data=data, meta_label='cluster_label')
frequencyFeatures['age'] = data.groupby('sample_id')['age'].first()
ages = frequencyFeatures['age']
frequencyFeatures = frequencyFeatures.drop(columns=['age'])

if 'donor' in data.columns:
    groups = data.groupby('sample_id')['donor'].first()
else:
    groups = pd.Series(frequencyFeatures.index, index=frequencyFeatures.index)
```

You can now give `frequencyFeatures` and `ages` to any ML model. The `groups` variable can be used for stratified train-test splitting, ensuring that samples from the same donor are not split between training and testing sets. 

## Example Usage

We provide two example scripts that demonstrate how to use all the featurization methods (scLKME, frequency, pseudobulk++, classical pseudobulk) for age prediction:

1. [example_code_classification.py](https://github.com/CompCy-lab/microglia-aging-clock/blob/main/example_code_classification.py): This script shows how to use the different featurization methods to extract features from single-cell data and use them for age classification using Random Forest.

2. [example_code_regression.py](https://github.com/CompCy-lab/microglia-aging-clock/blob/main/example_code_regression.py): This script demonstrates how to use the featurization methods to extract features and perform age regression using Lasso regression.

These examples provide a guide on how to preprocess data, extract features using various methods, and apply machine learning techniques for age prediction.

