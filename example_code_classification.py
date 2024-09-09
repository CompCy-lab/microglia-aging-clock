import os
import sys
import math
import random
from pathlib import Path
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import sclkme
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from microglia_aging_clock.clock import *

# Set random seed
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

# #### Accuracy
def accuracy(y_true, y_pred):
    """
    Function to calculate accuracy
    -> param y_true: list of true values
    -> param y_pred: list of predicted values
    -> return: accuracy score

    """
    # Intitializing variable to store count of correctly predicted classes
    correct_predictions = 0
    for yt, yp in zip(y_true, y_pred):

        if yt == yp:

            correct_predictions += 1
    #returns accuracy
    return correct_predictions / len(y_true)

def prepare_data(adata_merge):
    data = pd.DataFrame(adata_merge.X, columns=adata_merge.var_names, index=adata_merge.obs_names)
    data['sample_id'] = adata_merge.obs['sample_id'].copy()
    data['cluster_label'] = adata_merge.obs['leiden'].copy()
    data['age'] = adata_merge.obs['age'].copy()
    
    if 'donor' in adata_merge.obs.columns:
        data['donor'] = adata_merge.obs['donor'].copy()
    
    return data


# Run sclKME
def run_sclkme(adata_merge):
    sclkme.tl.sketch(adata_merge, n_sketch=1500, use_rep="X", key_added="X_512", method="kernel_herding")
    adata_sketch = adata_merge[adata_merge.obs['X_512_sketch']]
    X_anchor = adata_sketch.X.copy()
    sclkme.tl.kernel_mean_embedding(adata_merge, partition_key="sample_id", X_anchor=X_anchor, use_rep='X')
    
    sample_obs = adata_merge.obs[["sample_id", "age"]].groupby(by="sample_id", sort=False).agg(lambda x: x[0])
    adata_sample = ad.AnnData(adata_merge.uns['kme']['sample_id_kme'], obs=sample_obs)
    
    x = pd.DataFrame(adata_sample.X, index=adata_sample.obs.index)
    y = pd.Series(adata_sample.obs['age'], index=adata_sample.obs.index)
    y = np.array(y)
    
    if 'donor' in adata_merge.obs.columns:
        groups = adata_merge.obs.groupby('sample_id')['donor'].first()
    else:
        groups = pd.Series(x.index, index=x.index)
    
    return x, y, groups

# Run frequency_feats
def pred_on_frequency(adata_merge):
    data = prepare_data(adata_merge)
    FreqFeat = frequency_feats(data=data, meta_label='cluster_label')
    FreqFeat['age'] = data.groupby('sample_id')['age'].first()
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
    x, y = pseudobulk_plus_plus(data=data, adataObj=adata_merge, cluster='cluster_label', label='age', numFeat=numFeat)
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
    pb_f_hvg['age'] = data.groupby('sample_id')['age'].first()
    y = pb_f_hvg['age']
    x = pb_f_hvg.drop(columns=['age'])
    
    if 'donor' in data.columns:
        groups = data.groupby('sample_id')['donor'].first()
    else:
        groups = pd.Series(x.index, index=x.index)
    
    return x, y, groups


def run_classification(x, y, groups, output_filename, n_iterations=200):
    accuracies = []

    for _ in range(n_iterations):
        unique_groups = list(set(groups))
        train_groups = random.sample(unique_groups, int(0.8 * len(unique_groups)))
        
        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g not in train_groups]

        # Handle both DataFrame and NumPy array inputs for x
        if isinstance(x, pd.DataFrame):
            X_train, X_test = x.iloc[train_indices], x.iloc[test_indices]
        else:  # Assume NumPy array
            X_train, X_test = x[train_indices], x[test_indices]

        # Handle both Series and NumPy array inputs for y
        if isinstance(y, pd.Series):
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        else:  # Assume NumPy array
            y_train, y_test = y[train_indices], y[test_indices]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        rf_model = RandomForestClassifier(n_estimators=50, max_features="sqrt")
        rf_model.fit(X_train_scaled, y_train)

        predictions = rf_model.predict(X_test_scaled)
        acc = accuracy(y_test, predictions)
        accuracies.append(acc)

    results_df = pd.DataFrame({'Accuracy': accuracies})
    results_df.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")


def process_dataset(adata_path, dataset_name):
    adata_merge = sc.read_h5ad(adata_path)
    print(adata_merge)
    adata_merge.obs_names_make_unique()

    methods = [
        ('PB++25', lambda: run_pseudobulk_plus_plus(adata_merge, 25)),  # Pseudobulk++ with 25 cells
        ('Freq', lambda: pred_on_frequency(adata_merge)),               # Frequency-based prediction
        ('PB++50', lambda: run_pseudobulk_plus_plus(adata_merge, 50)),  # Pseudobulk++ with 50 cells
        ('sclkme', lambda: run_sclkme(adata_merge)),                    # scLKME method
        ('PBHVG', lambda: run_pseudobulk_hvg(adata_merge))              # Pseudobulk with highly variable genes
    ]


    for method_name, method_func in methods:
        print(f"Running {method_name} for {dataset_name}")
        x, y, groups = method_func()

        # Ensure x is a DataFrame and y is a Series
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x, index=groups.index)
        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=groups.index)

        run_classification(x, y, groups, f'{dataset_name}_{method_name}_classification.csv')


if __name__ == "__main__":
    process_dataset('./Hammond_adata_merge_with_donors.h5ad', 'Hammond')
    process_dataset('./Buckley_adata_merge.h5ad', 'Buckley')
    process_dataset('./Kracht_adata_merge_with_donors.h5ad', 'Kracht')

    print("Classification completed for all datasets.")