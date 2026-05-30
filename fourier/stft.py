import numpy as np
import xarray as xr
from scipy.signal import periodogram

# def windowed_periodogram(x, fs, window_size):
#     """
#     Compute windowed periodograms of a time series.

#     Parameters
#     ----------
#     x : array-like
#         1D time series
#     fs : float
#         Sampling rate (Hz)
#     window_size : int
#         Window size in samples

#     Returns
#     -------
#     xr.DataArray
#         DataArray with dims ('time', 'frequency')
#         - time: center of each window (in seconds)
#         - frequency: frequency bins (Hz)
#     """
#     x = np.asarray(x)
#     n = len(x)

#     if window_size > n:
#         raise ValueError("window_size must be <= length of time series")

#     # Number of full windows (non-overlapping)
#     n_windows = n // window_size

#     spectra = []
#     times = []

#     for i in range(n_windows):
#         start = i * window_size
#         end = start + window_size
#         segment = x[start:end]

#         # Compute periodogram
#         f, pxx = periodogram(segment, fs=fs)

#         spectra.append(pxx)

#         # Center time of the segment (in seconds)
#         center_sample = start + window_size / 2
#         times.append(center_sample / fs)

#     return xr.DataArray(
#             np.array(spectra).T,
#             dims=("f", "t"),
#             coords={
#                 "t": np.array(times),
#                 "f": f
#             },
#         )


def windowed_periodogram(x, fs, window_size, axis=-1):
    """
    Compute windowed periodograms of an nD array along a specified axis.

    Parameters
    ----------
    x : array-like
        Input nD array.
    fs : float
        Sampling rate (Hz).
    window_size : int
        Window size in samples.
    axis : int, optional
        Axis along which to compute the periodograms.
        Default is -1.

    Returns
    -------
    xr.DataArray
        DataArray containing the periodograms.

        Output dimensions are:
        - original dimensions except the analyzed axis
        - "window" : window index / time segment
        - "f" : frequency bins

        Coordinates:
        - "t" : center time of each window (seconds)
        - "f" : frequency bins (Hz)
    """
    x = np.asarray(x)

    axis = np.core.numeric.normalize_axis_index(axis, x.ndim)
    n = x.shape[axis]

    if window_size > n:
        raise ValueError("window_size must be <= length of selected axis")

    # Number of full non-overlapping windows
    n_windows = n // window_size

    # Truncate to full windows only
    trimmed_length = n_windows * window_size

    # Move target axis to the end
    x_moved = np.moveaxis(x, axis, -1)

    # Keep only complete windows
    x_moved = x_moved[..., :trimmed_length]

    # Reshape into (..., n_windows, window_size)
    new_shape = x_moved.shape[:-1] + (n_windows, window_size)
    x_windowed = x_moved.reshape(new_shape)

    # Compute periodogram along the last axis
    f, pxx = periodogram(x_windowed, fs=fs, axis=-1)

    # pxx shape:
    # (..., n_windows, n_freq)

    # Move frequency axis before window axis
    pxx = np.moveaxis(pxx, -1, -2)

    # Time coordinates (center of windows)
    times = (
        (np.arange(n_windows) * window_size + window_size / 2) / fs
    )

    # Build dimension names
    dims = [f"dim_{i}" for i in range(x.ndim)]
    original_dim = dims[axis]

    # Remove analyzed axis and append window/frequency dims
    out_dims = dims[:axis] + dims[axis + 1:] + ["f", "t"]

    # Build coordinates
    coords = {
        "f": f,
        "t": times,
    }

    return xr.DataArray(
        pxx,
        dims=out_dims,
        coords=coords,
        attrs={
            "fs": fs,
            "window_size": window_size,
            "analyzed_axis": original_dim,
        },
    )