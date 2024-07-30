import pandas as pd
import os
import anndata
import scanpy as sc
import numpy as np

def frequency_feats(data = None, meta_label = 'meta_label'):
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
    freq_feats = num.divide(denom, axis = 0)
    freq_feats.columns = ['freq_{}'.format(i) for i in freq_feats.columns]

    return freq_feats

def pseudobulk_plusplus(data = None, adataObj = None, cluster = None, label = None,numFeat = None):
    """
    Implements with fixed imporxtance per gene

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


    #first part is the ranking genes by overall importance
    adata_merge.uns['log1p']["base"] = None
    sc.tl.rank_genes_groups(adataObj, groupby="leiden", use_raw=False)
    dedf = sc.get.rank_genes_groups_df(adata_merge, group = None)
    print(dedf)
    dedf.iloc[:,2] = dedf.iloc[:,2].abs()
    dedf['scores'] = dedf['scores'].map(lambda x: math.log(x))

    temp = dedf.sort_values(by = 'scores', ascending = False)
    temp = temp.iloc[:numFeat]
    dedf = temp
    #print(dedf)
    dedf = dedf[['names','scores']]
    print(dedf)

    row_ind = 0
    col_ind = 0

    pb_feat_mat = np.zeros((data['sample_id'].nunique(),data[cluster].nunique()))

	samp_label = []

    for samp in data['sample_id'].unique():

        print(samp)

        for_label = data[((data['sample_id'] == samp))]

        samp_label.append(for_label['age'].value_counts().idxmax())

        print(for_label['age'].value_counts().idxmax())

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