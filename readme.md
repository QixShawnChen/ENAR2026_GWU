# 🧬 Integrating Multi-Omics Data for Breast Cancer Survival Prediction  
### *(ENAR 2026 Project)*

> A machine learning–based survival analysis study integrating clinical variables, gene expression, and DNA methylation data from TCGA breast cancer cohorts.

---

## 🔍 Overview

This project analyzes **TCGA breast cancer data** obtained from **cBioPortal**  
(Cerami et al., 2012) to investigate the following key question:
Does molecular data improve survival prediction beyond clinical variables alone?

### 🎯 Objectives

- Evaluate whether **gene expression** and **DNA methylation** improve survival prediction
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


