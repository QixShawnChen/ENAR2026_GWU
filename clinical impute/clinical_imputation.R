#upload datasets
data_clinical<-read.csv('/Users/mac1/Desktop/clinical.csv')
continue_set<-c('LYMPH_NODES_EXAMINED_POSITIVE', #Number of lymph nodes positive
              'NPI', #Nottingham prognostic index
              'AGE_AT_DIAGNOSIS', #Age at Diagnosis
              'TUMOR_SIZE', #Tumor Size
              'TMB_NONSYNONYMOUS') #TMB nonsynonymous
multi_set<-c('CELLULARITY', #Tumor Content
                 'COHORT', #Cohort
                 'CLAUDIN_SUBTYPE', #Pam50 Claudin low subtype
                 'THREEGENE', #X3 Gene classifier subtype
                 'HISTOLOGICAL_SUBTYPE', #Tumor Other Histologic Subtype
                 'CANCER_TYPE_DETAILED', #Cancer Type Detailed
                 'GRADE', #Neoplasm Histologic Grade
                 'ONCOTREE_CODE', #Oncotree Code
                 'TUMOR_STAGE' #Tumor Stage
)
binary_set<-c('CHEMOTHERAPY', #Chemotherapy
              'HORMONE_THERAPY', #Hormone Therapy
              'INFERRED_MENOPAUSAL_STATE', #Inferred Menopausal State
              'PR_STATUS', #PR Status
              'LATERALITY', #Primary Tumor Laterality
              'RADIO_THERAPY', #Radio Therapy
              'BREAST_SURGERY', #Type of Breast Surgery
              'ER_STATUS', #ER Status
              'HER2_STATUS' #HER2 Status
              )

data_clinical$CELLULARITY[data_clinical$CELLULARITY=='']<-NA
data_clinical$THREEGENE[data_clinical$THREEGENE=='']<-NA
data_clinical$LATERALITY[data_clinical$LATERALITY=='']<-NA
data_clinical$HISTOLOGICAL_SUBTYPE[data_clinical$HISTOLOGICAL_SUBTYPE=='']<-NA
data_clinical$BREAST_SURGERY[data_clinical$BREAST_SURGERY=='']<-NA


#imputation
dat<-data_clinical[,c(2,4,7:29)]
dat[continue_set] <- lapply(dat[continue_set], as.numeric)
dat[categorical_set] <- lapply(dat[categorical_set],factor)
dat[binary_set] <- lapply(dat[binary_set],factor)

str(dat)


library(mice)
ini  <- mice(dat, maxit = 0, print = FALSE)
meth <- ini$method
pred <- ini$predictorMatrix
meth[continue_set]      <- "pmm"
meth[binary_set]   <- "logreg"
meth[binary_set] <- "polyreg"
meth[colSums(is.na(dat)) == 0] <- ""
imp <- mice(
  dat,
  method = meth,
  predictorMatrix = pred,
  m = 1,
  maxit = 20,
  seed = 0,
  print = TRUE
)

dat_imp1 <- complete(imp, 1)

sum(is.na(dat_imp1))     # 0 missing now
str(dat_imp1)  

data_clinical[,c(2,4,7:29)] <- dat_imp1

write.csv(
  data_clinical,
  file = "/Users/mac1/Desktop/clinical_imputed_1.csv",
  row.names = FALSE
)
