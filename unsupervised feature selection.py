# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# %%
merged_all = pd.read_csv("/Users/mac1/Desktop/DATAFEST/merged_1_4.csv")
# %%
merged_all.shape #we have 2509 merged samples
# %%
#Exclusion criteria 1:Subject miss living status/survival time
mask_survival = (
    merged_all['OS_MONTHS'].notna() &
    merged_all['OS_STATUS'].notna()
)
merged_survival = merged_all.loc[mask_survival].copy()
merged_survival.shape #1981 left, 528 excluded
# %%
# Exclusion criteria 2.Subject misses all gene expression and methylation data
genetic_block = merged_survival.iloc[:, 36:33827]
mask_genetic = ~genetic_block.isna().all(axis=1)
merged_survival = merged_survival.loc[mask_genetic].copy()
merged_survival.shape # 1980 left, 1 excluded ->final analytic dataset
# %%
#split the analytic into clinical and genetic
merged_clinical=merged_survival.iloc[:,:35]
merged_genetic = merged_survival.iloc[:, [0] + list(range(36, merged_all.shape[1]))]
merged_clinical.count() #1 indicator, 34 clinical features
# %%
merged_clinical.to_csv("/Users/mac1/Desktop/merged_clinical.csv", index=False)
# %%
#split the genetic dataset into mythelation and mRNA
col_idx=merged_genetic.columns.get_loc("RERE_mrna")
methylation_genetic = merged_genetic.iloc[:, 0:col_idx]
mRNA_genetic=merged_genetic.iloc[:, [0] + list(range(col_idx, merged_genetic.shape[1]))]
# %%
#distribution of mythelation
X_meth = methylation_genetic.iloc[:, 1:].astype(float)
feature_means = X_meth.mean(axis=0)
feature_vars  = X_meth.var(axis=0)

plt.figure()
plt.hist(feature_means, bins=100)
plt.title("Distribution of Methylation Feature Means")
plt.xlabel("Mean")
plt.ylabel("Count")
plt.show()

plt.figure()
plt.hist(feature_vars, bins=100)
plt.title("Distribution of Methylation Feature Variances")
plt.xlabel("Variance")
plt.ylabel("Count")
plt.show()
# %%
#distribution of mRNA
X_mRNA = mRNA_genetic.iloc[:, 1:].astype(float)
feature_means = X_mRNA.mean(axis=0)
feature_vars  = X_mRNA.var(axis=0)

plt.figure()
plt.hist(feature_means, bins=100)
plt.title("Distribution of mRNA Feature Means")
plt.xlabel("Mean")
plt.ylabel("Count")
plt.show()

plt.figure()
plt.hist(feature_vars, bins=100)
plt.title("Distribution of mRNA Feature Variances")
plt.xlabel("Variance")
plt.ylabel("Count")
plt.show()
# %% [markdown]
# Clearly, the scales differ from mRNA and methylation. Before we perform feature selection on the entire genetic variable sets, we have to first standardize them to have them on the same scale.
# %%
X_mRNA.shape #n=20603 initial mRNA
# %%
X_meth.shape #n=13187 initial methylation
# %%
#standardization
X_meth_z = StandardScaler().fit_transform(X_meth)
X_mRNA_z = StandardScaler().fit_transform(X_mRNA)
X_genetic_z = np.hstack([X_meth_z, X_mRNA_z])
feature_means = X_genetic_z.mean(axis=0)
feature_vars  = X_genetic_z.var(axis=0)
print("Vars :", np.percentile(feature_vars,  [0, 1, 5, 50, 95, 99, 100]))
# %% [markdown]
# Oh no, there must be something wrong with variance standardization. The potential issue could be:
# 1.Some columns are entirely NaN
# 2.Some columns are constant (zero variance) after standardization
# 3.Some columns still contain non-numeric values
# %%
# Count NaNs per column
#X_genetic_df = pd.DataFrame(X_genetic)
X_meth_df = pd.DataFrame(X_meth)
X_mRNA_df = pd.DataFrame(X_mRNA)
nan_counts_meth = X_meth_df.isna().sum()
nan_counts_mRNA = X_mRNA_df.isna().sum()
missing_pct_meth=X_meth_df.isna().mean() * 100
missing_pct_mRNA=X_mRNA_df.isna().mean() * 100
print("Columns with NaN in methylation:", (nan_counts_meth > 0).sum())
print("Columns with NaN in mRNA:", (nan_counts_mRNA > 0).sum())
print("Percentage of missing in methylation",missing_pct_meth.describe())
print("Percentage of missing in mRNA",missing_pct_mRNA.describe())
# %% [markdown]
# Clearly, even we don't have feature of 100% missing, but there's some over 30%. If it's >30% missing, the data quality is bad and the imputation becomes non-imformative.
# %%
# Exclusion criteria 3: Features with high percentage of missingness (>30%)
keep_mask = missing_pct_meth <= 30 #only methylation
X_meth_df_comp =X_meth_df.loc[:, keep_mask]
X_meth_df_comp.shape #we exclude 3716 features
# account for 0
# %%
# Recall the variance plots, we may have some features have zero-variance / near-zero-variance features, let's do a quick check here
feat_var_meth  = X_meth_df_comp.var(axis=0)
feat_var_mRNA  = X_mRNA_df.var(axis=0)
print("\nPer-feature variance summary for methylation:")
print(feat_var_meth.describe(percentiles=[.05, 0.1,0.15,0.2,0.25,0.3,.5,.95, .99]))
print("\nPer-feature variance summary for mRNA:")
print(feat_var_mRNA.describe(percentiles=[.05, 0.1,0.15,0.2,0.25,0.3,.5,.95, .99]))
# %% [markdown]
# The variance after dropping >30% missing features is still very "zero-inflated"; If we fabricate and standardize those with small absolute variance, we may exaggerate some measurement error and give meaningless signal. We decide to drop bottom 20% with variance for both.
# %%
#Exclusion criteria 4: Features with extremely low variance (<15-20%)
var_raw_meth = X_meth_df_comp.var(axis=0, skipna=True)
cutoff_meth = var_raw_meth.quantile(0.2)
print("Variance cutoff:", cutoff_meth)
keep_meth = var_raw_meth > cutoff_meth
print("Removed features:", (~keep_meth).sum())
X_meth_df_var = X_meth_df_comp.loc[:, keep_meth]
X_meth_df_var.shape #we remove 1895 features (~0.00005)
# %%
var_raw_mRNA = X_mRNA_df.var(axis=0, skipna=True)
cutoff_mRNA = var_raw_mRNA.quantile(0.2)
print("Variance cutoff:", cutoff_mRNA)
keep_mRNA = var_raw_mRNA > cutoff_mRNA
print("Removed features:", (~keep_mRNA).sum())
X_mRNA_df_var = X_mRNA_df.loc[:, keep_mRNA]
X_mRNA_df_var.shape #we remove 4121 features (~0.025)
# %%
# Medium imputation for columns with missingness
X_meth_df_imp = X_meth_df_var.apply(lambda col: col.fillna(col.median()), axis=0)
X_mRNA_df_imp = X_mRNA_df_var.apply(lambda col: col.fillna(col.median()), axis=0)
# %%
#Standardization based on imputed X
# Methylation
scaler_meth = StandardScaler()
X_meth_df_imp_z = pd.DataFrame(
    scaler_meth.fit_transform(X_meth_df_imp),
    columns=X_meth_df_imp.columns,
    index=X_meth_df_imp.index
)

# mRNA
scaler_mrna = StandardScaler()
X_mRNA_df_imp_z = pd.DataFrame(
    scaler_mrna.fit_transform(X_mRNA_df_imp),
    columns=X_mRNA_df_imp.columns,
    index=X_mRNA_df_imp.index
)
# %% [markdown]
# Okay, let's go ahead for correlation pruning. Here, we choose to use pure feature–feature correlation pruning, becasue joint Cox models confounds relevance with redundancy; penalized Cox alone becomes not stable under extreme correlation.
# %%
#pure featurdef correlation_prune(X, threshold=0.5):
def correlation_prune_df(X: pd.DataFrame, threshold: float = 0.50):
    """
    Returns:
      keep_mask: np.ndarray[bool] of length p
      kept_cols: list of column names kept
    """
    Xv = X.to_numpy()  # keep column names separately
    n = Xv.shape[1]
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep[i]:
            continue
        xi = Xv[:, i]
        for j in range(i + 1, n):
            if keep[j]:
                corr = np.corrcoef(xi, Xv[:, j])[0, 1]
                if abs(corr) > threshold:
                    keep[j] = False

    kept_cols = X.columns[keep].tolist()
    return keep, kept_cols

# %%
keep_meth, meth_cols = correlation_prune_df(X_meth_df_imp_z, threshold=0.50)
X_meth_pruned = X_meth_df_imp_z.loc[:, meth_cols]
X_meth_pruned.shape # we exclude 1573 one of >0.5 correlated pairs
# %%
X_meth_pruned.to_csv("/Users/mac1/Desktop/X_meth_pruned.csv", index=False)
# %%
keep_mrna, mrna_cols = correlation_prune_df(X_mRNA_df_imp_z, threshold=0.50)
X_mRNA_pruned = X_mRNA_df_imp_z.loc[:, mrna_cols]
X_mRNA_pruned.shape # we exclude 6588 one of >0.5 correlated pairs
# %%
X_mRNA_pruned.to_csv("/Users/mac1/Desktop/X_mRNA_pruned.csv", index=False)
# %% [markdown]
# By the point, we have done feature selection solely based on genetic data structures (Unsupervised feature pre-selection). Let's then move on for Supervised survival feature selection to select most joinly relevant features with survival by Elastic Net Cox.
