# load packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ===============================
# Reproducibility / Random Seed
# ===============================
import os
import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
# numpy
np.random.seed(SEED)
# PyTorch
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# read dataset
X_train_df = pd.read_csv("/Users/liyajie/Downloads/Program/ENAR 2026 DataFest/data/Test_Train_15_85/X_train_integrate_1_imp.csv")
X_test_df = pd.read_csv("/Users/liyajie/Downloads/Program/ENAR 2026 DataFest/data/Test_Train_15_85/X_test_integrate_1_imp.csv")
Y_train_df = pd.read_csv("/Users/liyajie/Downloads/Program/ENAR 2026 DataFest/data/Test_Train_15_85/y_train_integrate_1.csv")
Y_test_df =  pd.read_csv("/Users/liyajie/Downloads/Program/ENAR 2026 DataFest/data/Test_Train_15_85/y_test_integrate_1.csv")
gwas = pd.read_csv("/Users/liyajie/Downloads/Program/ENAR 2026 DataFest/data/genomics_selected_gwas_ensembl.csv")



# outcome: 'OS_MONTHS', 'OS_STATUS'
# categorical predictors: 'CELLULARITY', 'CHEMOTHERAPY', 'COHORT',
# 'HORMONE_THERAPY', 'INFERRED_MENOPAUSAL_STATE', 'CLAUDIN_SUBTYPE', 'THREEGENE','PR_STATUS', 'LATERALITY',
# 'RADIO_THERAPY', 'HISTOLOGICAL_SUBTYPE', 'BREAST_SURGERY', 'CANCER_TYPE_DETAILED', 'ER_STATUS', 'HER2_STATUS',
# 'GRADE', 'ONCOTREE_CODE', 'TUMOR_STAGE'
# continuous: 'LYMPH_NODES_EXAMINED_POSITIVE', 'NPI', 'AGE_AT_DIAGNOSIS', 'TUMOR_SIZE','TMB_NONSYNONYMOUS',

# time to event/censor
dura_train = Y_train_df["OS_MONTHS"].astype(float).values
dura_test =  Y_test_df["OS_MONTHS"].astype(float).values

# event
event_map = {"0:LIVING": 0, "1:DECEASED": 1}
event_train = Y_train_df["OS_STATUS"].map(event_map).astype(int).values
event_test = Y_test_df["OS_STATUS"].map(event_map).astype(int).values

# predictors
feature_cols = ['CELLULARITY', 'CHEMOTHERAPY', 'COHORT','HORMONE_THERAPY', 'INFERRED_MENOPAUSAL_STATE',
                'CLAUDIN_SUBTYPE', 'THREEGENE','PR_STATUS', 'LATERALITY','RADIO_THERAPY', 'HISTOLOGICAL_SUBTYPE',
                'BREAST_SURGERY', 'CANCER_TYPE_DETAILED', 'ER_STATUS', 'HER2_STATUS','GRADE',
                'ONCOTREE_CODE', 'TUMOR_STAGE','LYMPH_NODES_EXAMINED_POSITIVE', 'NPI', 'AGE_AT_DIAGNOSIS',
                'TUMOR_SIZE','TMB_NONSYNONYMOUS']


X = X_train_df.loc[:,feature_cols]
X_test = X_test_df.loc[:,feature_cols]


X_train, X_val, dur_train, dur_val, ev_train, ev_val = train_test_split(
    X, dura_train, event_train, test_size=0.2, random_state=42, stratify=event_train)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

# predictors
# select categorical variables
cat_col = ['CELLULARITY', 'CHEMOTHERAPY', 'COHORT','HORMONE_THERAPY',
           'INFERRED_MENOPAUSAL_STATE', 'CLAUDIN_SUBTYPE', 'THREEGENE',
           'PR_STATUS', 'LATERALITY','RADIO_THERAPY', 'HISTOLOGICAL_SUBTYPE',
           'BREAST_SURGERY', 'CANCER_TYPE_DETAILED', 'ER_STATUS', 'HER2_STATUS',
           'GRADE', 'ONCOTREE_CODE', 'TUMOR_STAGE']
# one hot categorical variables
X_train_cat = pd.get_dummies(X_train[cat_col], drop_first=False)
X_val_cat = pd.get_dummies(X_val[cat_col], drop_first=False).reindex(columns=X_train_cat.columns, fill_value=0)
X_test_cat = pd.get_dummies(X_test[cat_col], drop_first=False).reindex(columns=X_train_cat.columns, fill_value=0)

# select numeric variables
num_col = ['LYMPH_NODES_EXAMINED_POSITIVE', 'NPI', 'AGE_AT_DIAGNOSIS', 'TUMOR_SIZE','TMB_NONSYNONYMOUS']
num_col = num_col

# scale numeric variable
scaler = StandardScaler()
X_num_train = scaler.fit_transform(X_train[num_col].to_numpy().astype(np.float32))
X_num_val = scaler.transform(X_val[num_col].to_numpy().astype(np.float32))
X_num_test = scaler.transform(X_test[num_col].to_numpy().astype(np.float32))

# concatenate predictors
X_train = np.hstack([X_num_train, X_train_cat.to_numpy().astype(np.float32)])
X_val   = np.hstack([X_num_val,   X_val_cat.to_numpy().astype(np.float32)])
X_test   = np.hstack([X_num_test,   X_test_cat.to_numpy().astype(np.float32)])


print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(dur_train.shape)
print(dur_val.shape)
print(dura_test.shape)
print(ev_train.shape)
print(ev_val.shape)
print(event_test.shape)

# Convert to tuples expected by pycox
y_train = (dur_train, ev_train)
y_val = (dur_val, ev_val)

import torchtuples as tt
from pycox.evaluation import EvalSurv

# Define a callback to compute C-index each epoch
class CIndexCallback(tt.callbacks.Callback):
    def __init__(self, model, X_train, y_train, X_val, y_val):
        self.model = model
        self.X_train = X_train
        self.dur_train, self.ev_train = y_train
        self.X_val = X_val
        self.dur_val, self.ev_val = y_val
        self.train_cindex = []
        self.val_cindex = []

    def on_epoch_end(self):
        # CoxPH/DeepSurv 需要 baseline hazards 才能 predict_surv_df
        self.model.compute_baseline_hazards()

        # train c-index
        surv_tr = self.model.predict_surv_df(self.X_train)
        ev_tr = EvalSurv(surv_tr, self.dur_train, self.ev_train, censor_surv='km')
        self.train_cindex.append(ev_tr.concordance_td())

        # val c-index
        surv_va = self.model.predict_surv_df(self.X_val)
        ev_va = EvalSurv(surv_va, self.dur_val, self.ev_val, censor_surv='km')
        self.val_cindex.append(ev_va.concordance_td())



# Neural net architecture
in_features = X_train.shape[1]
net = tt.practical.MLPVanilla(
    in_features=in_features,
    num_nodes=[128],
    out_features=1,
    batch_norm=True,
    dropout=0.5
)

# CoxPH model (DeepSurv)
optimizer = tt.optim.Adam(lr=1e-2, weight_decay=1e-3)
model = CoxPH(net, optimizer)

# callbacks
cindex_cb = CIndexCallback(model, X_train, y_train, X_val, y_val)
callbacks = [tt.callbacks.EarlyStopping(patience=25), cindex_cb]

# Train
batch_size = 128
epochs = 200
log = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    val_data=(X_val, y_val)
)

hist = log.to_pandas()

print(hist.columns)
print(hist.head())

# loss
plt.figure()
plt.plot(hist["train_loss"], label="Train loss")
plt.plot(hist["val_loss"], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# c-index
plt.figure()
plt.plot(cindex_cb.train_cindex, label="Train C-index")
plt.plot(cindex_cb.val_cindex, label="Val C-index")
plt.xlabel("Epoch")
plt.ylabel("C-index")
plt.legend()
plt.show()


# 训练结束后，用 test 评估
model.compute_baseline_hazards()

surv_test = model.predict_surv_df(X_test)
ev_test = EvalSurv(surv_test, dura_test, event_test, censor_surv='km')

cindex_test = ev_test.concordance_td()
print("Test C-index:", cindex_test)


#
# import numpy as np
# from lifelines import KaplanMeierFitter
# import matplotlib.pyplot as plt
#
# # 预测风险分数（越大风险越高）
# risk_test = model.predict(X_test).reshape(-1)
#
# # 按中位数分成高/低风险
# cut = np.median(risk_test)
# high = risk_test >= cut
# low  = risk_test < cut
#
# kmf = KaplanMeierFitter()
#
# plt.figure()
# kmf.fit(dura_test[low], event_observed=event_test[low], label="Low risk")
# kmf.plot_survival_function()
#
# kmf.fit(dura_test[high], event_observed=event_test[high], label="High risk")
# kmf.plot_survival_function()
#
# plt.xlabel("Time (months)")
# plt.ylabel("Survival probability")
# plt.title("Test set: KM by predicted risk group")
# plt.show()




import scipy.integrate
if not hasattr(scipy.integrate, "simps"):
    scipy.integrate.simps = scipy.integrate.simpson

import torchtuples as tt
from pycox.models import MTLR
from pycox.evaluation import EvalSurv

# Make sure baseline hazards are computed (needed for predict_surv_df)
model.compute_baseline_hazards()

# Predict survival curves on test
surv_test = model.predict_surv_df(X_test)   # DataFrame: index=time grid, columns=subjects
ev_test = EvalSurv(surv_test, dura_test, event_test, censor_surv='km')

# ---- Choose an evaluation time grid for IBS ----
# Use overlapping time range where you have data; avoid extremes
t_min = np.percentile(dura_test, 5)
t_max = np.percentile(dura_test, 95)

# If your surv_test index is already a good grid, restrict to [t_min, t_max]
times = surv_test.index.values
times = times[(times >= t_min) & (times <= t_max)]

# Need at least 2 time points
if len(times) < 2:
    raise ValueError("Not enough time points to compute IBS. Check your time grid and t_min/t_max.")

# ---- IBS (Integrated Brier Score) ----
ibs = ev_test.integrated_brier_score(times)
print("Test IBS:", ibs)

# (Optional) Brier score curve over time (for plotting)
bs = ev_test.brier_score(times)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(times, bs)
# plt.xlabel("Time")
# plt.ylabel("Brier score")
# plt.title("Test Brier score curve")
# plt.show()