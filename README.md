# Microglia Aging Clock from Computational Featurization Techniques
<p>
  <img src="https://github.com/CompCy-lab/microglia-aging-clock/blob/main/microglia_img.png?raw=True" width="350" />
</p>

##Overview

##Installation

```
git clone https://github.com/CompCy-lab/microglia-aging-clock
```

Then enter the microglia-aging-clock working directory as,

```
cd microglia-aging-clock
```

##Example for frequency features

Read in h5ad file containing single-cell data in annData format. We assumed cells have clusters with key `leiden`, sample ids indicating the samples they came from stored with key `sample_id`, and associated ages of the sample they came from stored with key `age`.

```python
import anndata
import os
adata_merge = anndata.read_h5ad(os.path.join('data','Hammond_adata_merge.h5ad'))
```
Given the keys described above with metadata about the cells, prepare a pandas data frame to input to the frequency function

```python
data_input = pd.DataFrame(adata_merge.X, columns = adata_merge.var_names, index = adata_merge.obs_names)
data_input['sample_id'] = adata_merge.obs['sample_id'].copy()
data_input['cluster_label'] = adata_merge.obs['leiden'].copy()
data_input['age'] = adata_merge.obs['age'].copy()
```
Extract frequency features and an associated list of ages.

```python
frequencyFeatures = frequency_feats(data = data_input, meta_label = 'cluster_label')
age_list = list(data_input.groupby(['sample_id'])['age'].agg(pd.Series.mode))
```


