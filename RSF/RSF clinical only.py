# ============================================================
# Random Survival Forest with 5-fold CV hyperparameter tuning
# ============================================================

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, integrated_brier_score

# --------------------
# Reproducibility
# --------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# --------------------
# Paths
# --------------------

X_TRAIN_PATH = "/Users/liyajie/Downloads/Program/ENAR 2026 DataFest/data/Test_Train_15_85/X_train_integrate_1_imp.csv"
X_TEST_PATH  = "/Users/liyajie/Downloads/Program/ENAR 2026 DataFest/data/Test_Train_15_85/X_test_integrate_1_imp.csv"
Y_TRAIN_PATH = "/Users/liyajie/Downloads/Program/ENAR 2026 DataFest/data/Test_Train_15_85/y_train_integrate_1.csv"
Y_TEST_PATH  = "/Users/liyajie/Downloads/Program/ENAR 2026 DataFest/data/Test_Train_15_85/y_test_integrate_1.csv"



# --------------------
# Load data
# --------------------
X_train_df = pd.read_csv(X_TRAIN_PATH)
X_test_df  = pd.read_csv(X_TEST_PATH)
Y_train_df = pd.read_csv(Y_TRAIN_PATH)
Y_test_df  = pd.read_csv(Y_TEST_PATH)

# --------------------
# Outcomes
# --------------------
dur_train = Y_train_df["OS_MONTHS"].astype(float).values
dur_test  = Y_test_df["OS_MONTHS"].astype(float).values

event_map = {"0:LIVING": 0, "1:DECEASED": 1}
ev_train = Y_train_df["OS_STATUS"].map(event_map).astype(int).values
ev_test  = Y_test_df["OS_STATUS"].map(event_map).astype(int).values

# --------------------
# Feature columns
# --------------------
cat_cols = [
    'CELLULARITY','CHEMOTHERAPY','COHORT','HORMONE_THERAPY',
    'INFERRED_MENOPAUSAL_STATE','CLAUDIN_SUBTYPE','THREEGENE',
    'PR_STATUS','LATERALITY','RADIO_THERAPY','HISTOLOGICAL_SUBTYPE',
    'BREAST_SURGERY','CANCER_TYPE_DETAILED','ER_STATUS','HER2_STATUS',
    'GRADE','ONCOTREE_CODE','TUMOR_STAGE'
]

num_cols = [
    'LYMPH_NODES_EXAMINED_POSITIVE','NPI','AGE_AT_DIAGNOSIS',
    'TUMOR_SIZE','TMB_NONSYNONYMOUS'
]



feature_cols = cat_cols + num_cols

# --------------------
# Preprocessing (fit ONCE on training)
# --------------------
Xtr_raw = X_train_df[feature_cols].copy()
Xte_raw = X_test_df[feature_cols].copy()

Xtr_cat = pd.get_dummies(Xtr_raw[cat_cols], drop_first=False)
Xte_cat = pd.get_dummies(Xte_raw[cat_cols], drop_first=False)\
              .reindex(columns=Xtr_cat.columns, fill_value=0)

scaler = StandardScaler()
Xtr_num = scaler.fit_transform(Xtr_raw[num_cols].astype(np.float32))
Xte_num = scaler.transform(Xte_raw[num_cols].astype(np.float32))

X_tr = np.hstack([Xtr_num, Xtr_cat.to_numpy(np.float32)])
X_te = np.hstack([Xte_num, Xte_cat.to_numpy(np.float32)])

# --------------------
# Structured survival arrays
# --------------------
def to_struct_y(dur, ev):
    return np.array(
        list(zip(ev.astype(bool), dur.astype(float))),
        dtype=[('event','?'),('time','<f8')]
    )

y_tr = to_struct_y(dur_train, ev_train)
y_te = to_struct_y(dur_test, ev_test)

# --------------------
# 5-fold CV function
# --------------------
def rsf_cv(X, dur, ev, params, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = []

    for tr_idx, va_idx in skf.split(X, ev):
        rsf = RandomSurvivalForest(
            n_estimators=params["n_estimators"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            n_jobs=-1,
            random_state=SEED
        )
        rsf.fit(X[tr_idx], to_struct_y(dur[tr_idx], ev[tr_idx]))

        risk = rsf.predict(X[va_idx])
        cidx = concordance_index_censored(
            ev[va_idx].astype(bool),
            dur[va_idx],
            risk
        )[0]
        scores.append(cidx)

    return np.mean(scores), np.std(scores)

# --------------------
# Hyperparameter grid
# --------------------
param_grid = [
    {"n_estimators": 500,  "min_samples_split": 20, "min_samples_leaf": 50, "max_features": 0.05},
    {"n_estimators": 500,  "min_samples_split": 50, "min_samples_leaf": 50, "max_features": 0.05},
    {"n_estimators": 500,  "min_samples_split": 20, "min_samples_leaf": 30, "max_features": 0.05},

    {"n_estimators": 1000, "min_samples_split": 20, "min_samples_leaf": 30, "max_features": 0.05},
    {"n_estimators": 1000, "min_samples_split": 20, "min_samples_leaf": 30, "max_features": 0.1},
    {"n_estimators": 1000, "min_samples_split": 10, "min_samples_leaf": 20, "max_features": 0.05},
    {"n_estimators": 1000, "min_samples_split": 10, "min_samples_leaf": 20, "max_features": "sqrt"},

    {"n_estimators": 1500, "min_samples_split": 10, "min_samples_leaf": 10, "max_features": 0.05},]

# --------------------
# Run CV tuning
# --------------------
print("\n5-fold CV tuning:\n")
best_score = -np.inf
best_params = None

for p in param_grid:
    mean_c, sd_c = rsf_cv(X_tr, dur_train, ev_train, p)
    print(f"{p} → C-index = {mean_c:.4f} ± {sd_c:.4f}")

    if mean_c > best_score:
        best_score = mean_c
        best_params = p

print("\nBest parameters:", best_params)
print("Best CV C-index:", best_score)

# --------------------
# Refit final RSF on ALL training data
# --------------------
rsf_final = RandomSurvivalForest(
    **best_params,
    n_jobs=-1,
    random_state=SEED
)
rsf_final.fit(X_tr, y_tr)

# --------------------
# Test C-index
# --------------------
risk_test = rsf_final.predict(X_te)
c_test = concordance_index_censored(
    ev_test.astype(bool),
    dur_test,
    risk_test
)[0]
print("\nTest C-index:", c_test)

# --------------------
# Test IBS
# --------------------
t_min, t_max = np.percentile(dur_test, [5, 95])
times = np.linspace(t_min, t_max, 100)

sf = rsf_final.predict_survival_function(X_te, return_array=False)
S_test = np.row_stack([fn(times) for fn in sf])

ibs = integrated_brier_score(y_tr, y_te, S_test, times)
print("Test IBS:", ibs)