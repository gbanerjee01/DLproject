import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import gzip
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

# TODO: add gzip code

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

result = pd.merge(diagnoses_icd, admissions, on='SUBJECT_ID')
output = result['HOSPITAL_EXPIRE_FLAG']
result.drop(columns=['HOSPITAL_EXPIRE_FLAG', 'DIAGNOSIS', 'DEATHTIME', 'DISCHARGE_LOCATION'])

onehot_encoder = preprocessing.OneHotEncoder(sparse=True)
onehot_encoded = onehot_encoder.fit_transform(result.astype(str).to_numpy().reshape(-1,len(result.columns)))
y = result['HOSPITAL_EXPIRE_FLAG'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(onehot_encoded, y, test_size=0.2, random_state=0)

# MODELING and LEARNING
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1).fit(X_train, y_train)

# INFERENCE
y_pred = clf.predict(X_test)

precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)
auc = metrics.auc(recall, precision)

print('Score:', clf.score(X_test, y_test))
print('AUC: {0:0.2f}'.format(auc))