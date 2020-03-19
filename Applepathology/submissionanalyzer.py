import pandas as pd
import numpy as np
import seaborn as sns
import os

submissionfile = "/home/alex//Fastdata/plant-pathology-2020/mysbms.csv"

sdf = pd.read_csv(submissionfile)
print(sdf)
for i in range(len(sdf)):
    if sdf.loc[i, "rust"] > 0.9 and sdf.loc[i, "scab"] > 0.9:
        sdf.loc[i, "multiple_diseases"] = 1.0
    else:
        sdf.loc[i, "multiple_diseases"] = 0

print(sdf)

sdf.to_csv("/home/alex//Fastdata/plant-pathology-2020/mysbms_edited.csv",index=False)