import pandas as pd
import numpy as np
import math


def z_normalize(ts):
    """
    Calculate the z-normalized time series by subtracting the mean and
    dividing by the standard deviation along a given axis.

    Parameters
    ----------
    ts : numpy.ndarray
        Time series.
    
    Returns
    -------
    norm_ts : numpy.ndarray
        The z-normalized time series.
    """

    norm_ts = (ts - np.mean(ts, axis=0)) / np.std(ts, axis=0)

    return norm_ts


def sliding_window(ts, window, step=1):
    """
    Extract subsequences from time series using sliding window.

    Parameters
    ----------
    ts : numpy.ndarray
        Time series.

    window : int
        Size of the sliding window.

    step : int
        Step of the sliding window.

    Returns
    -------
    subs_matrix : numpy.ndarray
        Matrix of subsequences.
    """
    
    n = ts.shape[0]
    N = math.ceil((n-window+1)/step)

    subs_matrix = np.zeros((N, window))

    for i in range(N):
        start_idx = i*step
        end_idx = start_idx + window
        subs_matrix[i] = ts[start_idx:end_idx]

    return subs_matrix

def read_ts(filepath, sep=None):
    """
    Read time series from a text file into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the file.
    sep : str, optional
        Separator (space by default).

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with time series (first column: label, others: series).
    """
    # Для файлов с множественными пробелами используем delim_whitespace
    df = pd.read_csv(filepath, delim_whitespace=True, header=None)
    return df

def random_walk(*args, **kwargs):
    """
    Generate a random walk time series.

    Parameters
    ----------
    length : int
        Length of the time series.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ts : numpy.ndarray
        Random walk time series.
    """
    length = kwargs.get('length', 100)
    seed = kwargs.get('seed', None)
    if seed is not None:
        np.random.seed(seed)
    steps = np.random.normal(loc=0, scale=1, size=length)
    ts = np.cumsum(steps)
    return ts
