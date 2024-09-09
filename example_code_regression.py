import os
import sys
import math
import random
from pathlib import Path
from itertools import cycle
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import sclkme
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from microglia_aging_clock.clock import *


# Set random seed
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

def prepare_data(adata_merge):
    """ Prepare data from AnnData object for featurization methods. """
    data = pd.DataFrame(adata_merge.X, columns=adata_merge.var_names, index=adata_merge.obs_names)
    data['sample_id'] = adata_merge.obs['sample_id'].copy()
    data['cluster_label'] = adata_merge.obs['leiden'].copy()
    data['age'] = adata_merge.obs['age'].copy()
    data['age_continuous'] = adata_merge.obs['age_continuous'].astype(float).copy()
    
    if 'donor' in adata_merge.obs.columns:
        data['donor'] = adata_merge.obs['donor'].copy()
    
    return data



# Run sclKME
def run_sclkme(adata_merge):
    # Sketching Part
    #sketching reduces dataset size and creates a smaller, representative subset of the data using kernel herding
    sclkme.tl.sketch(adata_merge, n_sketch=1500, use_rep="X", key_added="X_512", method="kernel_herding")
    
    adata_sketch = adata_merge[adata_merge.obs['X_512_sketch']]
    print(adata_sketch.shape)
    # get the precomputed anchor cells
    X_anchor = adata_sketch.X.copy()
    # Kernel Mean Embedding (KME)
    #Computes the kernel mean embeddings for the samples using pre-computed anchor cells
    sclkme.tl.kernel_mean_embedding(adata_merge, partition_key="sample_id", X_anchor=X_anchor, use_rep='X')
    
    # Aggregating Sample Observations
    sample_obs = adata_merge.obs[["sample_id", "age_continuous"]].groupby(by="sample_id", sort=False).agg(lambda x: x[0])
    
    # Create AnnData object for samples
    adata_sample = ad.AnnData(adata_merge.uns['kme']['sample_id_kme'], obs=sample_obs)
    print(adata_sample.obs)
    
    # Extract Features and Labels
    x = pd.DataFrame(adata_sample.X, index=adata_sample.obs.index)
    y = pd.Series(adata_sample.obs['age_continuous'], index=adata_sample.obs.index)
    
    # Determine groups based on donor information if available
    if 'donor' in adata_merge.obs.columns:
        groups = adata_merge.obs.groupby('sample_id')['donor'].first()
    else:
        groups = pd.Series(x.index, index=x.index)
    
    return x, y, groups


# Run frequency_feats
def pred_on_frequency(adata_merge):
    data = prepare_data(adata_merge)
    FreqFeat = frequency_feats(data=data, meta_label='cluster_label')
    FreqFeat['age'] = data.groupby('sample_id')['age_continuous'].first()
    y = FreqFeat['age']
    x = FreqFeat.drop(columns=['age'])
    
    if 'donor' in data.columns:
        groups = data.groupby('sample_id')['donor'].first()
    else:
        groups = pd.Series(x.index, index=x.index)
    
    return x, y, groups

# Run pseudobulk_plus_plus
def run_pseudobulk_plus_plus(adata_merge, numFeat):
    data = prepare_data(adata_merge)
    x, y = pseudobulk_plus_plus(data=data, adataObj=adata_merge, cluster='cluster_label', label='age_continuous', numFeat=numFeat, label_type='continuous')
    x = pd.DataFrame(x, index=data['sample_id'].unique())
    y = pd.Series(y, index=x.index)
    
    if 'donor' in data.columns:
        groups = data.groupby('sample_id')['donor'].first()
    else:
        groups = pd.Series(x.index, index=x.index)
    
    return x, y, groups

# Run pseudobulk_hvg
def run_pseudobulk_hvg(adata_merge):
    data = prepare_data(adata_merge)
    pb_f_hvg = pseudobulk_hvg(data=data, meta_label='cluster_label')
    pb_f_hvg['age'] = data.groupby('sample_id')['age_continuous'].first()
    y = pb_f_hvg['age']
    x = pb_f_hvg.drop(columns=['age'])
    
    if 'donor' in data.columns:
        groups = data.groupby('sample_id')['donor'].first()
    else:
        groups = pd.Series(x.index, index=x.index)
    
    return x, y, groups



# Lasso Regression 
def run_cross_validation(x, y, groups, output_filename='results.csv', scale=True):
    # Ensure x, y, and groups are properly aligned
    x = pd.DataFrame(x)
    y = pd.Series(y, index=x.index)
    groups = pd.Series(groups, index=x.index)

    cv = LeaveOneGroupOut() if len(np.unique(groups)) < len(groups) else LeaveOneOut()
    split_data = cv.split(x, y, groups)

    outs, gt, sample_ids = [], [], []
    scaler = StandardScaler()

    for train_index, test_index in split_data:
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if scale:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        lasso_cv = LassoCV(cv=5, max_iter=100000)
        lasso_cv.fit(X_train, y_train)
        lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=100000)
        lasso.fit(X_train, y_train)
        yy = lasso.predict(X_test)

        outs.extend(yy)
        gt.extend(y_test)
        sample_ids.extend(y_test.index)

    results_df = pd.DataFrame({
        'Sample_ID': sample_ids,
        'Ground_Truth': gt,
        'Predictions': outs,
        'Group': groups[sample_ids].values
    })

    results_df.to_csv(output_filename, index=False)
    print(f"Data saved to {output_filename}.")

    return results_df


def process_dataset(adata_path, dataset_name):
    adata_merge = sc.read_h5ad(adata_path)
    adata_merge.obs_names_make_unique()

    # Run all methods
    methods = [
        ('Freq', lambda: pred_on_frequency(adata_merge)),               # Frequency
        ('PB++25', lambda: run_pseudobulk_plus_plus(adata_merge, 25)),  # Pseudobulk++25 
        ('PB++50', lambda: run_pseudobulk_plus_plus(adata_merge, 50)),  # Pseudobulk++50
        ('sclkme', lambda: run_sclkme(adata_merge)),                    # scLKME
        ('PBHVG', lambda: run_pseudobulk_hvg(adata_merge))              # Pseudobulk with highly variable genes
    ]

    for method_name, method_func in methods:
        print(f"Running {method_name} for {dataset_name}")
        x, y, groups = method_func()
        run_cross_validation(x, y, groups, f'{dataset_name}_{method_name}_lasso_regression.csv')


if __name__ == "__main__":
    process_dataset('./Hammond_adata_merge_with_donors.h5ad', 'Hammond')
    process_dataset('./Buckley_adata_merge.h5ad', 'Buckley')
    process_dataset('./Kracht_adata_merge_with_donors.h5ad', 'Kracht')

    print("Regression completed for all datasets.")