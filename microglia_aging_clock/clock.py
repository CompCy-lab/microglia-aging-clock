import pandas as pd
import os
import anndata
import scanpy as sc
import numpy as np
import math # need to import math


#### Frequency
def frequency_feats(data=None, meta_label='meta_label'):
    """
    Computes frequency features for each sample based on the proportion of cells 
    assigned to each metacluster.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing single-cell information. Must include columns for
        'sample_id' and the metacluster labels.
    meta_label : str, optional
        Name of the column in 'data' containing metacluster label IDs.
        Default is 'meta_label'.

    Returns
    -------
    freq_feats : pandas.DataFrame
        DataFrame containing frequency features for each sample. Each row represents
        a sample, and each column represents the proportion of cells in a specific
        metacluster, named as 'freq_{metacluster_id}'.

    Notes
    -----
    - The function assumes that 'sample_id' is a column in the input DataFrame.
    - Samples with no cells in a particular metacluster will have a frequency of 0.
    - The sum of frequencies for each sample (row) will equal 1.
    """

    num = data.groupby(['sample_id', meta_label]).size().unstack().fillna(0)
    denom = num.sum(1)
    freq_feats = num.divide(denom, axis=0)
    freq_feats.columns = ['freq_{}'.format(i) for i in freq_feats.columns]
    return freq_feats


#### Pseudobulk++
def pseudobulk_plus_plus(data = None, adataObj = None, cluster = None, label = None,numFeat = None, label_type='categorical'):
    """
    Implements pseudobulk analysis with fixed importance per gene.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data containing single-cell information.
    adataObj : AnnData
        AnnData object used to produce the data.
    cluster : str
        Name of the column in 'data' which stores cluster information.
    label : str
        Name of the column in 'data' which stores label information.
        This can be either categorical or continuous, as specified by label_type.
    numFeat : int
        Number of top features to consider for importance calculation.
    label_type : str, optional
        Type of the label column. Can be 'categorical' or 'continuous'.
        Default is 'categorical'.

    Returns
    -------
    pb_feat_mat : numpy.ndarray
        Sample-level features matrix. Shape is (n_samples, n_clusters).
    samp_label : list
        Sample-level labels. For categorical labels, this will be the most frequent
        label per sample. For continuous labels, this will be the mean value per sample.
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
    Computes functional features by summing gene expression within metaclusters for each sample.
    This function is designed to work with highly variable genes (HVGs) but can be used with any gene expression data.

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataframe containing gene expression data. It should include:
        - Numeric columns for gene expression values
        - A 'sample_id' column to identify different samples
        - A column specified by 'meta_label' for metacluster information
        Note: The input can be pre-filtered to include only highly variable genes if desired.

    meta_label : str, optional
        Name of the column containing metacluster label IDs. These labels group cells 
        into metaclusters based on biological or technical criteria.
        Default is 'meta_label'.

    Returns
    -------
    pseudobulk_feats : pandas.DataFrame
        A dataframe containing pseudobulk features for each sample. Each row represents 
        a sample, and each column represents the summed expression of a gene within a 
        specific metacluster, named as '{gene}_exp_{metacluster}'.

    Raises
    ------
    ValueError
        If the required columns ('sample_id' and the specified 'meta_label') are missing from the input data.

    Notes
    -----
    - The function sums expression values for each gene across all cells within each 
      metacluster for each sample.
    - Only numeric columns are considered for expression summation.
    - Missing values in the resulting pseudobulk features are filled with 0.
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