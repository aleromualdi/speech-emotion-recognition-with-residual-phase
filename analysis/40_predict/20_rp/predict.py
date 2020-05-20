'''
In this script I measure the performance of the RPCCs-based CNN model on the
test set.
'''
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import librosa
import keras
from keras.models import model_from_json
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

sys.path.append('../../../')
from aux import load_audio_from_path
from residualphase import residual_phase

"""
Loading validation data
"""
x_valcnn = np.loadtxt(
    '../../30_train_cnn_model/20_rp/output/x_valcnn.csv', delimiter=',')
y_val = np.loadtxt(
    '../../30_train_cnn_model/20_rp/output/y_val.csv', delimiter=',')

x_valcnn = x_valcnn.reshape((192, 259, 1))
y_val = y_val.reshape((192, 2))

"""
Loading test data
"""
test_df = pd.read_csv('../../10_split/output/test_df.csv')
del test_df['Unnamed: 0']

"""
Loading the trained model
"""
json_file = open('../../30_train_cnn_model/20_rp/output/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(
    "../../30_train_cnn_model/20_rp/output/model_checkpoint.h5")

print("Loaded model from disk.")

"""
Evaluate model on validation data
"""
opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)

loaded_model.compile(
    loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

val_score = loaded_model.evaluate(x_valcnn, y_val, verbose=0)

print()
print("%s: %.2f%%" % (loaded_model.metrics_names[1], val_score[1] * 100))

"""
Predicting emotions on the test data
"""

rpccs_vec = []
for i in tqdm(range(len(test_df))):
    y, sr = load_audio_from_path('../../../data/' + test_df.path[i])
    # calculate residual phase
    rp_y = residual_phase(y)
    rpccs = np.mean(librosa.feature.mfcc(y=rp_y, sr=sr, n_mfcc=13), axis=0)
    rpccs_vec.append(rpccs)

df_rpccs = pd.DataFrame(rpccs_vec, test_df.label).reset_index()#.fillna(0)
ntest_df = pd.DataFrame(rpccs_vec).values

lb = LabelEncoder()
val_labels = np_utils.to_categorical(lb.fit_transform(test_df.label))
ntest_df = np.expand_dims(ntest_df, axis=2)

preds = loaded_model.predict(ntest_df, batch_size=16, verbose=1).argmax(axis=1)
predictions = (lb.inverse_transform((preds)))
preddf = pd.DataFrame({'predictedvalues': predictions})

actual = val_labels.argmax(axis=1)
actualvalues = (lb.inverse_transform((actual)))

actualdf = pd.DataFrame({'actualvalues': actualvalues})
finaldf = actualdf.join(preddf)

"""
Actual vs Predicted emotions
"""

finaldf.to_csv('./output/predictions.csv', index=False)

y_true = finaldf.actualvalues
y_pred = finaldf.predictedvalues

acc_scure = accuracy_score(y_true, y_pred) * 100
print('Accuracy score =', np.round(acc_scure, 2))
# Accuracy score = 68.75

f1_score = f1_score(y_true, y_pred, average='macro') * 100
print('F1 score =', np.round(f1_score, 2))
# F1 score = 68.47

"""
Confusion matrix
"""
c = confusion_matrix(y_true, y_pred)

class_names = ['sad', 'happy']


fig = plt.figure(figsize=(10,7))

df_cm = pd.DataFrame(c, index=class_names, columns=class_names)

heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

heatmap.yaxis.set_ticklabels(
    heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=13)
heatmap.xaxis.set_ticklabels(
    heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=13)

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout(True)
plt.savefig('output/confusion_matrix.png')
# plt.show()
