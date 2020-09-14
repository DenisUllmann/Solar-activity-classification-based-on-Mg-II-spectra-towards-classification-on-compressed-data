import os, tools
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

THIS_DIR_PATH = Path(os.path.abspath(__file__)).parent
npzfile_path = os.path.join(THIS_DIR_PATH,'packeddata')

#load the dataset
columns = ['label','dataarray']
df = pd.DataFrame(columns=columns)
filenames = glob(npzfile_path+'/*')
for file in tqdm(filenames):
    label = file[44:46]
    if os.path.isdir(file):
        npyfilenames = glob(file+'/*')
        for npyfile in npyfilenames:
            if npyfile.endswith('.npy'):
                data = np.load(npyfile)
                yindex, timeindex = data.shape
                for i in range(timeindex):
                    temp = []
                    for j in range(yindex//2-75,yindex//2+75):
                          temp.append(data[j,i])
                    df.loc[len(df)] = [label, temp]

df2 = shuffle(df)
X_train, X_test, Y_train, Y_test = train_test_split(df2['dataarray'], df2['label'], test_size=0.2)

#train the model
model = xgb.XGBClassifier(gamma=1,
                          learning_rate=0.01,
                          max_depth=18,
                          n_estimators=147,
                          random_state=42)

model.fit(np.array(list(X_train)), Y_train, verbose=10)
Y_pred = model.predict(np.array(list(X_test)))
print(Y_pred)
accuracy = accuracy_score(Y_test, Y_pred)
print(accuracy)
tools.plot_confusion_matrix(Y_test, Y_pred,normalize=True)

'''
# cross-validation 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits=2, random_state=7)
results = cross_val_score(model, np.array(list(X_train)), Y_train, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
'''
