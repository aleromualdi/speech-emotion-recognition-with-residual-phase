'''
In this script I calculate the mel-frequency cepstral coefficients (MFCCs) and
the residual phase cepstral coefficients (RPCCs) from each speech files of
the discovery set.
I perform data augmentation on the speech files.
'''

import sys
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../../')

from aux import load_audio_from_path
from aux.augmentation import noise, pitch
from residualphase import residual_phase


"""
Reading discovery data
"""
df = pd.read_csv('../10_split/output/disc_df.csv')
del df['Unnamed: 0']

"""
Getting the features of audio files using librosa
"""

mfccs_vec = []
mfccs_noise_vec = []
mfccs_pitch_vec = []

rpccs_vec = []
rpccs_noise_vec = []
rpccs_pitch_vec = []

for i in tqdm(range(len(df))):
    y, sr = load_audio_from_path('../../data/' + df.path[i])
    # augment data
    y_noise = noise(y)
    y_pitch = pitch(y, sr)
    #calculate residual phase
    yrp = residual_phase(y)
    yrp_noise = residual_phase(y_noise)
    yrp_pitch = residual_phase(y_pitch)

    # compute mfccs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_noise = librosa.feature.mfcc(y=y_noise, sr=sr, n_mfcc=13)
    mfcc_pitch = librosa.feature.mfcc(y=y_pitch, sr=sr, n_mfcc=13)
    # compute rpccs
    rpcc = librosa.feature.mfcc(y=yrp, sr=sr, n_mfcc=13)
    rpcc_noise = librosa.feature.mfcc(y=yrp_noise, sr=sr, n_mfcc=13)
    rpcc_pitch = librosa.feature.mfcc(y=yrp_pitch, sr=sr, n_mfcc=13)

    mfccs_vec.append(np.mean(mfcc, axis=0))
    mfccs_noise_vec.append(np.mean(mfcc_noise, axis=0))
    mfccs_pitch_vec.append(np.mean(mfcc_pitch, axis=0))

    rpccs_vec.append(np.mean(rpcc, axis=0))
    rpccs_noise_vec.append(np.mean(rpcc_noise, axis=0))
    rpccs_pitch_vec.append(np.mean(rpcc_pitch, axis=0))


"""
Creating MFCCs and RPCCs DataFrames
"""
# MFCC
df_mfccs = pd.DataFrame(mfccs_vec, df.emotion).reset_index().fillna(0)
df_mfccs_noise = pd.DataFrame(mfccs_noise_vec, df.emotion).reset_index().fillna(0)
df_mfccs_pitch = pd.DataFrame(mfccs_pitch_vec, df.emotion).reset_index().fillna(0)

combined_df_mfccs = pd.concat(
    [df_mfccs, df_mfccs_noise, df_mfccs_pitch], ignore_index=True)

# RPCC
df_rpccs = pd.DataFrame(rpccs_vec, df.emotion).reset_index().fillna(0)
df_rpccs_noise = pd.DataFrame(rpccs_noise_vec, df.emotion).reset_index().fillna(0)
df_rpccs_pitch = pd.DataFrame(rpccs_pitch_vec, df.emotion).reset_index().fillna(0)

combined_df_rpccs = pd.concat(
    [df_rpccs, df_rpccs_noise, df_rpccs_pitch], ignore_index=True)

print()
print('combined_df_mfccs.shape', combined_df_mfccs.shape)
# (768, 260)
print('combined_df_rpccs.shape', combined_df_rpccs.shape)
# (768, 260)

"""
Output
"""
combined_df_mfccs.to_csv('output/mfccs_disc.csv')
combined_df_rpccs.to_csv('output/rpccs_disc.csv')
