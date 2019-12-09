import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gzip
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# PREPROCESSING
out_dir = "/mnt/c/Users/benga/Documents/Classes/Autumn 2019/project/mimic-iii-clinical-database-1.4"
filename = "DIAGNOSES_ICD.csv.gz"
d_icd = "{}/{}".format(out_dir, filename)
d_icd_csv = gzip.open(d_icd)
diagnoses_icd = pd.read_csv(d_icd_csv)

filename = "ADMISSIONS.csv.gz"
adm = "{}/{}".format(out_dir, filename)
adm_csv = gzip.open(adm)
admissions = pd.read_csv(adm_csv)

# both diagnoses_icd and admissions have SUBJECT_ID as a column; need to merge these together
result = pd.merge(diagnoses_icd, admissions[['SUBJECT_ID', 'HOSPITAL_EXPIRE_FLAG']], on='SUBJECT_ID')

# use linear regression on numpy arrays to try to predict the expire flag; one-hot encode ICD9_CODE b/c it's a categorical variable, avoid misrepresenting data
onehot_encoder = preprocessing.OneHotEncoder(sparse=True)
onehot_encoded = onehot_encoder.fit_transform(result['ICD9_CODE'].astype(str).to_numpy().reshape(-1,1))

y = result['HOSPITAL_EXPIRE_FLAG'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(onehot_encoded, y, test_size=0.2, random_state=0)

# MODELING and LEARNING
reg = LogisticRegression().fit(X_train, y_train)

# INFERENCE
y_pred = reg.predict(X_test)

print('Score:', reg.score(X_test, y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
