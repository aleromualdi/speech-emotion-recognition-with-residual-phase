import numpy as np
import librosa


def noise(data):
    """
    Adding add white noise in the background.
    """
    # you can take any distribution from
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
    noise_amp = 0.005 * np.random.uniform() * np.amax(data)
    data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
    return data


def pitch(data, sample_rate):
    """
    Wrapper of effects.pitch_shift function. It shifts the pitch randomly.
    """
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change =  pitch_pm * 2 * (np.random.uniform())
    data = librosa.effects.pitch_shift(
                                    data.astype('float64'),
                                    sample_rate,
                                    n_steps=pitch_change,
                                    bins_per_octave=bins_per_octave)
    return data
