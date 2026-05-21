import numpy as np
import xarray as xr
from scipy.signal import periodogram

def windowed_periodogram(x, fs, window_size):
    """
    Compute windowed periodograms of a time series.

    Parameters
    ----------
    x : array-like
        1D time series
    fs : float
        Sampling rate (Hz)
    window_size : int
        Window size in samples

    Returns
    -------
    xr.DataArray
        DataArray with dims ('time', 'frequency')
        - time: center of each window (in seconds)
        - frequency: frequency bins (Hz)
    """
    x = np.asarray(x)
    n = len(x)

    if window_size > n:
        raise ValueError("window_size must be <= length of time series")

    # Number of full windows (non-overlapping)
    n_windows = n // window_size

    spectra = []
    times = []

    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        segment = x[start:end]

        # Compute periodogram
        f, pxx = periodogram(segment, fs=fs)

        spectra.append(pxx)

        # Center time of the segment (in seconds)
        center_sample = start + window_size / 2
        times.append(center_sample / fs)

    return xr.DataArray(
            np.array(spectra).T,
            dims=("frequency", "time"),
            coords={
                "time": np.array(times),
                "frequency": f
            },
        )