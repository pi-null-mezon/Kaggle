import pandas as pd
import os
import shutil

# Train file location
trainfilename = "/home/alex/Fastdata/plant-pathology-2020/train.csv"
srcimagespath = "/home/alex/Fastdata/plant-pathology-2020/images"
targetpath    = "/home/alex/Fastdata/plant-pathology-2020/Train"


traindata = pd.read_csv(trainfilename)

classeslist = []
for name in traindata.columns:
    if name != "image_id":
        classeslist.append(name)
        try:
            os.mkdir(os.path.join(targetpath,name))
        except Exception as exc:
            # do nothing if directory already exists
            pass


for i in range(len(traindata)):
    for classname in classeslist:
        if traindata.iloc[i][classname] == 1:
            filename = traindata.iloc[i]["image_id"] + ".jpg"
            shutil.copyfile(os.path.join(srcimagespath,filename),
                            os.path.join(os.path.join(targetpath,classname),filename))
