import numpy as np
import scipy
import librosa



def _check_y(y):
    pass


def lp(y, order=16):
    """
    Uses librosa.core.lpc to calculate linear prediction coefficients
    via Burg’s method. Then use scipy.signal.lfilter to calculate the LP
    residual by inverse filtering.

    """

    a = librosa.lpc(y, order=order)

    return scipy.signal.lfilter([0] + -1 * a[1:], [1], y)


def lp_residual(y, order=16):
    """
    Calculate linear prediction residual from the linear prediction estimate.


    Parameters
    ----------
    order : int >0 (default=16), order of the linear filter

    Returns
    -------
    res : linear prediction residual

    """

    y_lp = lp(y, order)
    return y - y_lp


def residual_phase(y, lp_residual_order=16):
    """
    Calculate the residual phase of a wave signal.
    The residual phase is defined as the cosine of the phase function of the
    analytic signal derived from the linear prediction (LP) residual.

    Parameters
    ----------
    y : numpy array or list representing wave signal signal.

    Returns
    -------

    residual_phase: numpy array, residual phase.

    """

    _check_y(y)

    r = lp_residual(y, order=lp_residual_order)

    # calculates analytical signal of a signal y by Hilbert transform.
    r_a = scipy.signal.hilbert(r)

    return r / np.abs(r_a)
