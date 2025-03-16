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
All three datasets from our original study can be found in a [Zenodo repository](https://zenodo.org/records/12811383?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjAxMGRkODUzLTY0NjAtNGU5MC1hNzgzLWUwMzhmZTZlOTZlYiIsImRhdGEiOnt9LCJyYW5kb20iOiI1OWE5ZDVmYmVjMmIzNjUxNzUxNTllZGMzNGYyMjgwNSJ9.rB0clZfO5rsht2svUh6rI5oQxgTUOrtHrzAME0Ms0PE9wk5_gl7lU5z0h9TCpMBsKs_5__psMC4LQp7kYJySCQ). The examples below use the [Hammond dataset, Hammond et al., Immunity 2019](https://www.sciencedirect.com/science/article/pii/S1074761318304850?via%3Dihub). 

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

## Comprehensive Characterization of Major Microglia Subtypes Across Studies
|Subtype                                |Description                                              |Upregulated Genes                                                                                                                                    |Downregulated Genes                        |Physiological/Pathological Context                                                   |References                                                                         |
|---------------------------------------|---------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|-------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
|Homeostatic (Homeos1)                  |Primary homeostatic population with surveillant functions|P2ry12, Cx3cr1, Tmem119, Sall1, Fcrls, Hexb, Csf1r, Tgfbr1, Mertk, Itgb5, Sparc, Gpr34, Siglech, Olfml3, Selplg, P2ry13, Adgrb1, Kbtbd12, Rasgef1c   |Itgax, Apoe, Cd74, Il1b, Ccl3, Ccl4, Mki67 |Normal healthy brain; dominant in adult brain                                        |Hammond et al. (2019), Masuda et al. (2019), Galatro et al. (2017)                 |
|Homeostatic-GRID2+ (Homeos2)           |Specialized homeostatic subtype                          |Grid2, Prdm11, Adgrb3, Syndig1, Dscam, Med12l, Nav2, Rasgef1c, Tbc1d4, Usp6nl, Map4k4, Dock10, Garnl3, Tbc1d9, Tbc1d12, Evi5, Snx13, Rap1gap2        |Apoe, Cd74, Itgax, Il1b                    |Potential response to tau-bearing neurons; specialized homeostatic function          |Martins-Ferreira et al. (2025), Gerrits et al. (2021)                              |
|Pre-activated (Homeos3)                |Transitional homeostatic with early activation           |P2ry12, Cx3cr1, Serpine1, Peli2, Foxp2, Inpp5d, Zfp36l2, Card11, Id2, Nckap1l, Csf3r, Padi2, Prex1, Trpm2, Swap70, Gab1, Rgs1, Cd83, Dusp1, Fos      |Itgax, Apoe, Cd74                          |Transition state between homeostasis and activation; response to subtle perturbations|Martins-Ferreira et al. (2025)                                                     |
|Inflammatory DAM (Inflam.DAM)          |Classical disease-associated microglia                   |Apoe, Clec7a, Itgax, Lgals3, Spp1, Axl, Lpl, Cst7, Cd74, Tyrobp, Trem2, Cd9, Csf1, Cd63, Lag3, Cd68, Ccl3, Ccl4, Tmem163, Hamp, Hspa5                |P2ry12, Tmem119, Cx3cr1, Fcrls, Hexb, Csf1r|Alzheimer's disease, amyloid pathology, neurodegeneration, aging                     |Keren-Shaul et al. (2017), Hammond et al. (2019), Martins-Ferreira et al. (2025)   |
|Lipid DAM (Lipo.DAM)                   |DAM subtype specialized in lipid processing              |Gpnmb, Lgals3, Fabp5, Lgals1, Lilrb4a, Ccl9, Anxa5, Pld3, Cd9, Cd36, Lpl, Apoe, Trem2, Fabp4, Npc1, Npc2, Msr1, Plin2, Apoc1, Pparg, Mitf            |P2ry12, Tmem119                            |Phagocytosis of myelin debris, expanded in demyelinating contexts and AD             |Marschallinger et al. (2020), Hammond et al. (2019), Martins-Ferreira et al. (2025)|
|Ribosomal DAM 1 (Ribo.DAM1)            |DAM subtype with ribosomal function signature            |Syt1, Pcdh9, Lsamp, Cadm2, Calm1, Kcnip4, Pebp1, Rbfox1, Cntnap2, Rpl39, Dynll1, Ndufa4, mt-Atp6, Rps27, Dlg2, Rtn3, mt-Nd1, Atp5e, Rpl18a, Rps18    |P2ry12, Tmem119, Cx3cr1                    |Associated with increased translational activity; found in neurodegenerative contexts|Martins-Ferreira et al. (2025), Sun et al. (2023)                                  |
|Ribosomal DAM 2 (Ribo.DAM2)            |Second DAM with distinct ribosomal signature             |Plekha7, Mecom, Ooep, Atp5f1e, Fth1, Baiap2l1, Plekha6, Bdnf-as, Ftl, Naca2, Slc27a4, Apoo, Slc47a1, Znf90, Txnrd1, Xpo5, Rpl19, Tmsb10, Uba52, Tomm7|P2ry12, Tmem119, Cx3cr1                    |Similar to Ribo.DAM1 but with distinct ribosomal gene expression pattern             |Martins-Ferreira et al. (2025), Sun et al. (2023)                                  |
|Activated Response Microglia (ARM)     |Microglia with activated response profile                |H2-Ab1, H2-Eb1, Cd74, Dkk2, Gpnmb, Spp1, Apoe, Trem2, Tyrobp, Itgax, Lgals3, Cst7, Ctss                                                              |P2ry12, Tmem119, Cx3cr1                    |Activated in early stages of neurodegeneration; associated with AD risk genes        |Sala Frigerio et al. (2019), Sierksma et al. (2020)                                |
|Interferon-Responsive Microglia (IRM)  |Microglia with prominent interferon response             |Ifi27l2a, Ifitm3, Ccl12, Bst2, Ifit3, Isg15, Ifi204, Irf7, Cxcl10, Oasl2, Rtp4, B2m, Stat1, Isgf3g, Ifih1, Ifit1, Irf9, Irf1, Irf8, Zbp1, Ch25h      |P2ry12, Tmem119, Cx3cr1                    |Viral infections, sterile inflammation, neurodegeneration with type-I IFN            |Hammond et al. (2019), Martins-Ferreira et al. (2025)                              |
|MHCII-high                             |Microglia with high MHC class II expression              |H2-Ab1, H2-Eb1, Cd74, Ciita, H2-K1, B2m, Cd86, Il1b, Tnf, H2-Aa, Cd40, Nlrc5, Tap1, Tap2, Psmb8, Psmb9                                               |P2ry12, Trem2, Cx3cr1                      |Antigen presentation; increased in EAE, MS models, and aging                         |Mathys et al. (2017), Mrdjen et al. (2018)                                         |
|Proliferating                          |Proliferating microglia                                  |Mki67, Top2a, Cenpe, Mcm5, Birc5, Cdk1, Ccnb2, Pcna, Aurkb, Cenpf, Plk1, Cdca8, Cdca3, Cdc20, Ccna2, Ccnb1                                           |P2ry12, Trem2, Cx3cr1                      |Injury response, development, repopulation after depletion                           |Hammond et al. (2019)                                                              |
|Disease-Inflammatory Macrophages (DIMs)|Microglia-like cells from infiltrating monocytes         |Slc2a3, Cd83, Ccl3, Dusp1, Ch25h, Nampt, Dnajb1, Fos, Hspa1a, Srgn, Rhob, Hsph1, Hspa1b, Irak2, Hif1a, Rgs1, Btg2, Mcl1, Crem, Jun, Junb, Ier2       |P2ry12, Tmem119, Sall1                     |BBB disruption, traumatic injury, experimental autoimmune encephalomyelitis          |Silvin et al. (2022), Martins-Ferreira et al. (2025)                               |
|Border-Associated Macrophages (MAC)    |Macrophages in CNS borders                               |F13a1, Cd163, Mrc1, Ms4a7, Lyve1, Folr2, Stab1, Pf4, Cbr2, Cd209f, Cd209g, Cd163l1, Mgl2, Cd209d, Cd209a                                             |P2ry12, Cx3cr1, Tmem119                    |Meninges, choroid plexus, perivascular spaces; CNS-immune interface                  |Mrdjen et al. (2018), Utz et al. (2020)                                            |
|Axon Tract-associated Microglia (ATM)  |Microglia associated with axon tracts                    |Itgax, Igf1, Spp1, Gpnmb, Lgals3, Cd68, Cd9, Fabp5, Pld3, Ctsl, Lgals1, Lilrb4a, Cd63, Ctsb                                                          |P2ry12, Tmem119, Cx3cr1                    |White matter development, myelin development, corpus callosum                        |Hammond et al. (2019)                                                              |
|Embryonic Microglia                    |Microglia found during early development                 |Spi1, Cx3cr1, Fabp5, Mcm5, Ccnd1, Dab2, Mrc1, Ms4a7, Fn1, Csf1r, Cd4, Runx1, F13a1                                                                   |P2ry12, Tmem119, Sall1                     |Embryonic and early postnatal brain development (E14.5-P4)                           |Hammond et al. (2019)                                                              |
|Neonatal CD11c+ Microglia              |CD11c-expressing microglia in development                |Itgax, Igf1, Spp1, Gpnmb, Clec7a, Trem2, Lgals3, Cd68, Fabp5, Lpl, Csf1, Mrc1, Cd74                                                                  |P2ry12, Tmem119                            |Early postnatal development (P1-P8); synaptic pruning, brain wiring                  |Wlodarczyk et al. (2017), Benmamar-Badel et al. (2020)                             |
|Transitional IFN-DAM                   |Transitional state with IFN response and DAM activation  |Isg15, Ifi44l, Irf9, Ifitm3, Irf7, Zbp1, Cxcl10, Stat1, Mx1, Oasl1, Isg20, Ifit1, Ifit2, Ifit3, Apoe, Trem2                                          |P2ry12, Tmem119, Cx3cr1                    |Early phase of disease-associated transformation; transition state                   |Mathys et al. (2017)                                                               |
|Advanced DAM                           |Advanced stage of DAM activation with S100 expression    |H2-Ab1, H2-Eb1, Cd74, S100a6, S100a8, S100a9, Ccl6, Ccl9, Lgals3, Cd63, Itgax, Cybb, Trem2, Apoe, Tyrobp                                             |P2ry12, Tmem119, Cx3cr1, Isg15, Ifi44l     |Advanced neurodegeneration; Late stage AD, chronic inflammation                      |Mathys et al. (2017)                                                               |


