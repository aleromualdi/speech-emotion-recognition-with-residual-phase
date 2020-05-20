'''
In this script I split the data into 'discovery' and 'test' sets.
The discovery set will be further split into 'train' and 'validation' sets for
model training. The 'test' set will be used to measure the performance of the
trained models.
'''

import sys
import pandas as pd
sys.path.append('../../')
from aux.reader import read_data


"""
Reading data
"""
print('reading data...')
data_df = read_data('../../data/')

"""
Selecting subclasses
"""
data_df = data_df[data_df.gender=='male']

labels = ['happy', 'sad']
data_df = data_df[data_df.emotion.isin(labels)]

actors = data_df.actor.unique()
print(len(actors), actors)
# 12, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]

"""
Train/test splitting
"""
test_actors = actors[-2 :]

disc_df = data_df[~data_df.actor.isin(test_actors)]
test_df = data_df[data_df.actor.isin(test_actors)]

"""
Stats
"""
print()
print('len(disc_df)', len(disc_df))
print('len(test_df)', len(test_df))
# len(disc_df) 320
# len(test_df) 64

print()
print('Discovery set')
print(disc_df.groupby('emotion').count()['actor'])
# happy    160
# sad      160

print()
print('Test set')
print(test_df.groupby('emotion').count()['actor'])
# happy    32
# sad      32

"""
Saving to file
"""
disc_df.emotion = disc_df.emotion.replace(
    {'happy': 'male_positive', 'sad': 'male_negative'})
test_df.emotion = test_df.emotion.replace(
    {'happy': 'male_positive', 'sad': 'male_negative'})

disc_df.to_csv('output/disc_df.csv')
test_df.to_csv('output/test_df.csv')
