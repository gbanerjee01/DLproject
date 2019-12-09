import pandas as pd
import gzip

# PREPROCESSING
out_dir = "/mnt/c/Users/benga/Documents/Classes/Autumn 2019/project/mimic-iii-clinical-database-1.4"
filename = "ADMISSIONS.csv.gz"
adm = "{}/{}".format(out_dir, filename)
adm_csv = gzip.open(adm)
admissions = pd.read_csv(adm_csv)

admissions.to_csv('modded_csv.csv')

just_text_and_hef = admissions[['DIAGNOSIS', 'HOSPITAL_EXPIRE_FLAG']]
# hef = admissions['HOSPITAL_EXPIRE_FLAG']

# new = diags

print(just_text_and_hef.head())
just_text_and_hef.to_csv("just_text_and_hef.csv")

