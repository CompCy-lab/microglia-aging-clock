import pandas as pd
import os
import anndata
import scanpy as sc
import numpy as np
import math # need to import math


#### Frequency
def frequency_feats(data=None, meta_label='meta_label'):
    """
    Computes frequency features (proportion of cells assigned to a metacluster within a sample)

    Parameters
    ----------
    data: pandas.DataFrame
    meta_label: str
        string referencing column containing metacluster label ID

    Returns
    ----------
    freq_feats: pandas.DataFrame
        dataframe containing frequency features for each sample
    """

    num = data.groupby(['sample_id', meta_label]).size().unstack().fillna(0)
    denom = num.sum(1)
    freq_feats = num.divide(denom, axis=0)
    freq_feats.columns = ['freq_{}'.format(i) for i in freq_feats.columns]
    return freq_feats

#### Pseudobulk++
def pseudobulk_plus_plus(data = None, adataObj = None, cluster = None, label = None,numFeat = None, label_type='categorical'):
    """
    Implements with fixed importance per gene 

    Parameters
    ----------
    data: pandas.DataFrame
    adataObj : ann data object used to produce data
    cluster : str for name of column of data which stores clusters
    label: str for name of column of data which stores labels
    ----------
    pb_feat_mat: numpy array
        sample-level features
    samp_label: sample level labels
    """
    adataObj.uns['log1p']["base"] = None
    sc.tl.rank_genes_groups(adataObj, groupby="leiden", use_raw=False)
    dedf = sc.get.rank_genes_groups_df(adataObj, group=None)
    
    dedf.iloc[:, 2] = dedf.iloc[:, 2].abs()
    dedf['scores'] = dedf['scores'].map(lambda x: math.log(x))
    dedf = dedf.sort_values(by='scores', ascending=False).iloc[:numFeat][['names', 'scores']]

    row_ind = 0
    col_ind = 0

    pb_feat_mat = np.zeros((data['sample_id'].nunique(), data[cluster].nunique()))
    samp_label = []

    for samp in data['sample_id'].unique():

        for_label = data[((data['sample_id'] == samp))]
        if label_type == 'categorical':
            samp_label.append(for_label[label].value_counts().idxmax())
        else:  # continuous
            samp_label.append(for_label[label].astype(float).mean())
        samp_feat_vec = []
        for cl in data[cluster].unique():
            #select by sample and cluster
            sample_by_cl = data[((data['sample_id'] == samp) & (data[cluster] == cl))]
            if sample_by_cl.empty:
                sum_wa = 0
            else:
                feature_only = sample_by_cl[dedf['names']].to_numpy()
                importance_vec = dedf['scores'].to_numpy()
                weighted_avg = feature_only @ importance_vec
                sum_wa = weighted_avg.sum()
            samp_feat_vec += [sum_wa]

        pb_feat_mat[row_ind,:] = samp_feat_vec
        row_ind = row_ind + 1
    return pb_feat_mat, samp_label


#### Classical Pseudobulk of highly variable genes
def pseudobulk_hvg(data=None, meta_label='meta_label'):
    """
    Computes functional features (summed expression of metaclusters within a sample)
    of all genes in sample.

    Parameters
    ----------
    data: pandas.DataFrame
        The dataframe containing gene expression data [which could be filtered to include only highly variable genes if desired].
    meta_label: str
        String referencing column containing metacluster label ID, which groups cells into metaclusters based on some biological or technical criteria.
    
    Returns
    ----------
    pseudobulk_feats: pandas.DataFrame
        Dataframe containing functional features for each sample, where features are summed expressions of genes within each metacluster.
    """
    #print("Columns available:", data.columns)  # Debug: check columns
    # Select numeric data for summing and retain columns necessary for grouping
    numeric_data = data.select_dtypes(include=[np.number])
    grouping_columns = data[['sample_id', meta_label]]

    combined_data = numeric_data.join(grouping_columns)

    # Check if grouping columns are present
    if 'sample_id' not in combined_data.columns or meta_label not in combined_data.columns:
        raise ValueError("Required columns for grouping are missing.")

    # Group the data by 'sample_id' and the specified 'meta_label'
    d = combined_data.groupby(['sample_id', meta_label])

    summed_data = {}
    
    # Iterate through each group, summing the expression data
    for name, group in d:
        # Ensure to sum only the numeric expression data!
        summed_expression = group.select_dtypes(include=[np.number]).sum()
        summed_data[name] = summed_expression 

    pseudobulk_feats = pd.DataFrame.from_dict(summed_data, orient='index')  
    pseudobulk_feats = pseudobulk_feats.unstack().fillna(0)  

    pseudobulk_feats.columns = ['{}_exp_{}'.format(col[1], col[0]) for col in pseudobulk_feats.columns]

    return pseudobulk_feats