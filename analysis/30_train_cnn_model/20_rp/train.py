'''
In this script I will train a CNN model on speech data for classification of
speaker emotion based on RPCCs.
'''

import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import (Dense , Flatten, Dropout, Activation,
    BatchNormalization, Conv1D, MaxPooling1D)
from keras.callbacks import (ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, CSVLogger)


"""
Reading the rpccs of discovery data
"""
df_rpccs_disc = pd.read_csv('../../20_mfccs_rpccs/output/rpccs_disc.csv')
del df_rpccs_disc['Unnamed: 0']

"""
Splitting train/validation
"""
def split(X, y):

    SSS = StratifiedShuffleSplit(1, test_size=0.2, random_state=12)
    for train_index, test_index in SSS.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))

    return X_train, X_test, y_train, y_test


X_rpccs = df_rpccs_disc.drop(['emotion'], axis=1)
y_rpccs = df_rpccs_disc.emotion
X_train, X_val, y_train, y_val = split(X_rpccs, y_rpccs)

np.savetxt('output/y_val.csv', y_val.reshape(-1), delimiter=',')

"""
Defining the CNN model
"""

def CNN_model(X_input_shape):
    model = Sequential()
    model.add(Conv1D(256, 8, padding='same',input_shape=(X_input_shape, 1)))
    model.add(Activation('relu'))
    model.add(Conv1D(256, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    # 2-classes
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model

# Changing dimension for CNN model
x_traincnn = np.expand_dims(X_train, axis=2)
x_valcnn = np.expand_dims(X_val, axis=2)

print()
print('x_traincnn.shape', x_traincnn.shape)
print('x_valcnn.shape', x_valcnn.shape)
# x_traincnn.shape (768, 259, 1)
# x_valcnn.shape (192, 259, 1)

# save to csv file
# np.savetxt('output/x_traincnn.csv', x_traincnn.reshape(-1), delimiter=',')
np.savetxt('output/x_valcnn.csv', x_valcnn.reshape(-1), delimiter=',')

"""
Setting up model with Keras
"""
model = CNN_model(X_train.shape[1])

# Stochastic gradient descent optimizer
opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

"""
Defining Callbacks
"""
# saves the model best weight
mcp_save = ModelCheckpoint(
                        './output/model_checkpoint.h5',
                        save_best_only=True,
                        monitor='val_loss',
                        mode='min')

# Reduce learning rate when a metric has stopped improving.
lr_reduce = ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=10, min_lr=0.000001)

# stops the training once validation loss ceases to decreas
early = EarlyStopping(monitor="val_loss", mode="min", patience=5)

epoch_log_path = 'logs/train_epochs.csv'
if os.path.exists(epoch_log_path):
    os.remove(epoch_log_path)

# streams epoch results to a csv file.
csv_logger = CSVLogger(epoch_log_path, append=True, separator=';')

callbacks_list = [mcp_save, lr_reduce, csv_logger]

"""
Model Training
"""
cnnhistory = model.fit(
                    x_traincnn,
                    y_train,
                    batch_size=16,
                    epochs=500,
                    validation_data=(x_valcnn, y_val),
                    callbacks=callbacks_list)

"""
Saving the model
"""
model_json = model.to_json()
with open("./output/model.json", "w") as json_file:
    json_file.write(model_json)

"""
Plotting the train and validation lossed
"""
plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('./output/model_loss.png')
plt.show()
