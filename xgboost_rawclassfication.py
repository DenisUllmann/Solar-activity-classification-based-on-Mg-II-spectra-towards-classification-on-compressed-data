import os
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tools import plot_confusion_matrix

npzfile_path = "C:/Users/Denis/Documents/IRIS/iris_level_2C"
csv_path = "D:/Maks_IRIS_class_comp"
crossvalidate = False

columns = ['label','dataarray']
df = pd.DataFrame(columns=columns)
filenames = glob(os.path.join(npzfile_path,'*.npz'))
for file in tqdm(filenames):
    label = os.path.split(file)[-1][:2]
    if label in ['QS', 'SS', 'AR', 'PF', 'FL'] and file.endswith('.npz'):
        data = np.load(file)['data']
        rindex, timeindex, yindex, wavindex = data.shape
        for r in range(rindex):
            for i in range(timeindex):
                temp = []
                for j in range(yindex//2-75,yindex//2+75):
                      temp.append(data[r,i,j])
                df.loc[len(df)] = [label, temp]
df2 = shuffle(df)
df2.to_csv(os.path.join(csv_path, 'rawxgb_prepared_df.csv'),index=False)

'''
df['label'].value_counts().plot(kind = "barh")
plt.xlabel("Count")
plt.ylabel("Classes")
plt.savefig('classes_count.png')
plt.show()


df3 = pd.read_csv('prepered_df.csv')
print(df2[1000:1005])
print(df3[1000:1005])
print(df2['dataarray'].shape[1])
print(df3['dataarray'].shape[1])
'''

X_train, X_test, Y_train, Y_test = train_test_split(df2['dataarray'], df2['label'], test_size=0.2)
'''

bestaccuracy = 0.5
for md in range(18,50):
    for ne in range(81,153,3):
        model = xgb.XGBClassifier(gamma=1,
            learning_rate=0.01,
            max_depth=md,
            n_estimators=ne,
            random_state=42)
        model.fit(np.array(list(X_train)).reshape((len(X_train), -1)), Y_train,verbose = 10)
        Y_pred = model.predict(np.array(list(X_test)).reshape((len(X_test), -1)))
        accuracy = accuracy_score(Y_test, Y_pred)
        if accuracy>bestaccuracy:
            bestaccuracy = accuracy
            bestmd_n_ne = (md,ne)
        print('md=',md,'ne=',ne, "Accuracy: %.2f%%" % (accuracy * 100.0))
#print(confusion_matrix(Y_test, Y_pred))
print('Best acc is:', bestaccuracy, 'with md and ne:', bestmd_n_ne)

'''
model = xgb.XGBClassifier(gamma=1,
                          learning_rate=0.01,
                          max_depth=19,
                          n_estimators=185,
                          random_state=42)

if os.path.isfile('saved_xgb_lr0-01_md19_ne185.dat'):
    print('loading model..')
    model = pickle.load(open('saved_xgb_lr0-01_md19_ne185.dat', 'rb'))
else:
    print('training model..')
    model.fit(np.array(list(X_train)).reshape((len(X_train), -1)), Y_train, verbose=10)
    print('saving model..')
    pickle.dump(model, open('saved_xgb_lr0-01_md19_ne185.dat', 'wb'))

Y_pred = model.predict(np.array(list(X_test)).reshape((len(X_test), -1)))
#print(Y_pred)
accuracy = accuracy_score(Y_test, Y_pred)
print('accuracy', accuracy)
# disp = plot_confusion_matrix(model, np.array(list(X_test)).reshape((len(X_test), -1)), Y_test, normalize='true', cmap='Blues')
# disp.ax_.set_title('Normalized confusion matrix')
# disp.figure_.savefig('RAW_confursion_matrix_xgb_lr0-01_md19_ne185.png')
plot_confusion_matrix(Y_test, Y_pred, normalize=True, name='RAW_confursion_matrix_xgb_lr0-01_md19_ne185.png')

if crossvalidate:
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=2, random_state=7)
    results = cross_val_score(model, np.array(list(X_train)).reshape((len(X_train), -1)), Y_train, cv=kfold)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


