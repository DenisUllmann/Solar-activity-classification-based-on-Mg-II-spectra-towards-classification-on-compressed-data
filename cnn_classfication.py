import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

THIS_DIR_PATH = Path(os.path.abspath(__file__)).parent
npzfile_path = os.path.join(THIS_DIR_PATH,'packeddata')

columns = ['label','dataarray']
df = pd.DataFrame(columns=columns)

#load the dataset
filenames = glob(npzfile_path+'/*')
for file in tqdm(filenames):
    label = file[44:46]
    if label == 'AR':
        label2=0
    elif label == 'FL':
        label2=1
    elif label == 'PF':
        label2=2
    elif label == 'QS':
        label2=3
    elif label == 'SS':
        label2=4
    if os.path.isdir(file):
        npyfilenames = glob(file+'/*')
        for npyfile in npyfilenames:
            if npyfile.endswith('.npy'):
                data = np.load(npyfile)
                yindex, timeindex = data.shape
                for i in range(timeindex):
                    temp = []
                    for j in range(yindex//2-75,yindex//2+75):
                          temp.append(data[j,i]/53)
                    df.loc[len(df)] = [label2, temp]
df2 = shuffle(df)
#trasnform to categorical data
X_train, X_test, Y_train, Y_test = train_test_split(df['dataarray'], df['label'], test_size=0.2)
X_train = list(X_train)
X_test = list(X_test)
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

#build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D(filters=54, kernel_size=9, activation='relu', input_shape=(150,1)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(model.summary())

#train the model
model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=1)
_, accuracy = model.evaluate(X_test, Y_test, batch_size=32, verbose=1)
print(accuracy)
