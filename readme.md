# 🧬 Integrating Multi-Omics Data for Breast Cancer Survival Prediction  
### *(ENAR 2026 Project)*

> A machine learning–based survival analysis study integrating clinical variables, gene expression, and DNA methylation data from TCGA breast cancer cohorts.

---

## 🔍 Overview

This project analyzes **TCGA breast cancer data** obtained from **cBioPortal**  
(Cerami et al., 2012) to investigate the following key question:
Does molecular data improve survival prediction beyond clinical variables alone?

### 🎯 Objectives

- Evaluate whether integrating **gene expression** and **DNA methylation** improve survival prediction
- Compare **prior-informed** vs **data-driven** feature selection strategies
- Identify **top-performing feature combinations** with strongest predictive power
- Assess **risk stratification capability**
- Compare predictive reliability across **different model families**

---

## 📊 Data Source

- **Dataset**: TCGA Breast Cancer (BRCA) 
https://drive.google.com/drive/folders/1ovMbyHBFum6pNwyx0eK_I7Uqwuq_T8mx

- **Modalities**:
  - **Clinical variables**
  - **RNA expression**
  - **DNA methylation**


**Data_clinical_patient.txt**: Contains clinical information about patients, including diagnosis, treatments received, age at diagnosis, overall survival time, and survival status.

**Data_clinical_sample.txt**: Provides details about the tumor samples, such as the associated patient, tumor site, stage, and other sample-specific annotations.

**Data_methylation_promoters_rrbs.txt**: Contains gene-level methylation data across samples.

**Data_mrna_illumina_microarray.txt**: Includes normalized mRNA expression levels from Illumina microarrays.

**Data_mutation.txt**: Contains mutation data for the tumor samples.


All .csv files created can be accessed via https://drive.google.com/drive/u/1/folders/1OA4BzoFf-l_Ha2ERI9p8y5tUOg2aZfq1

---

## 🧪 Workflow

```text

Data Preparation
     │
     ▼
Feature Selection by 3 strategies
     │
     ▼
Model Training
     │
     ▼
Evaluation & Visualization
```


## 💻 Code
- **Data Preparation**

    * /ENAR2026/dataset_preparation.ipynb: The script works for combining Clinical datasets, mRNA, and DNA methylation by subject id.

    * /ENAR2026/clinical_impute/clinical feature.ipynb: The script works for listing and renaming clinical variables.

    * /ENAR2026/clinical_impute/clinical imputation.R: The script works for multiple imputation for clinical variables.

    * /ENAR2026/test_train_split.ipynb: The script works for combining, splitting, and imputing different feature sets.

- **Feature Selection Strategy 1**

    * /ENAR2026/unsupervised feature selection.py: The script works for unsupervised feature selection for Strategy 1.


- **Feature Selection Strategy 2 & 3**

    * /ENAR2026/Feature Selection Strategy 2&3 /1script_annotation_preparation_table.ipynb: 
    The script converts the Data_mutation.txt table into the required format for 'Ensembl' to annotate.

    * /ENAR2026/Feature Selection Strategy 2&3 /2clinival_signal_select_rsid.ipynb:
    The script selects the variants by impact size, functional consequences, and clinical signal.

    * /ENAR2026/Feature Selection Strategy 2&3 /3clinical_signal_select_hugo_symbol.ipynb:
    The script extracts the corresponding Hugo symbols of selected variants.

    * /ENAR2026/Feature Selection Strategy 2&3 /4gwas_hugo_selection.ipynb:
    The script extracts the significant variants from well-known breast cancer GWAS Summary Statistics ("https://www.ebi.ac.uk/gwas/studies/GCST90018799")  and extracts the corresponding Hugo symbols of the selected variants.

    * /ENAR2026/Feature Selection Strategy 2&3 /5merge_selected_hugo_to_df.ipynb:
    The scripts selects features of combined dataset by merging the selected Hugo symbols with the prefix of mRNA features and methylation features.

- **Model Training & Evaluation & Visualization (Elastic Net)**

    * /ENAR2026/EN/EN.ipynb: 
    The script writes a function for Elastic Net with cross-validation and functions to get Figure 2, including risk stratification, top feature bar charts and heatmaps, where the importance score of heatmap are stored in seperated txt file under /ENAR2026/EN.

- **Model Training & Evaluation & Visualization (Random Survival Forest)**

*The scripts below summarize the random survival forest analyses with 5-fold cross-validation across different case combinations.*

    * /ENAR2026/RSF/RSF clinical + case2.py:

    * /ENAR2026/RSF/RSF clinical only.py:

    * /ENAR2026/RSF/RSF clinical+case1.py:

    * /ENAR2026/RSF/RSF clinical+gwas+case1.py:

    * /ENAR2026/RSF/RSF clinical+gwas+case2.py:

    * /ENAR2026/RSF/RSF gwas+clinical.py:

- **Model Training & Evaluation & Visualization (DeepSurv)**

*The scripts below summarize the DeepSurv model tuning parameters based on validation performance across different case combinations.*

    * /ENAR2026/Deepsurv/DeepSurv clinical only.py:

    * /ENAR2026/Deepsurv/DeepSurv clinical+case1+gwas.py:

    * /ENAR2026/Deepsurv/DeepSurv clinical+case2+gwas.py:

    * /ENAR2026/Deepsurv/Deepsurv clinical + case1.py:

    * /ENAR2026/Deepsurv/Deepsurv clinical + case 2.py:

    * /ENAR2026/DeepsurvRSF/Deepsurv clinical+c1 mrna only.py:

    * /ENAR2026/DeepsurvRSF/Deepsurv clinical+case 2 mrna only.py:

    * /ENAR2026/DeepsurvRSF/Deepsurv gwas+clinical.py:
