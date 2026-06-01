import numpy as np
import xarray as xr
from scipy.signal import periodogram

def trim_to_windows_around_injection(x, t, t_inj, w):
    """
    Trim x and t so that an integer number of windows of duration w
    fit on both sides of the injection time.

    Parameters
    ----------
    x : array_like
        Data.

    t : array_like
        Time stamps (uniformly sampled).

    t_inj : float
        Injection time.

    w : float
        Window duration.

    Returns
    -------
    x_trim : ndarray
    t_trim : ndarray
    i_inj_trim : int
        Index of injection sample in trimmed arrays.
    """
    x = np.asarray(x)
    t = np.asarray(t)

    dt = np.median(np.diff(t))
    nwin = int(round(w / dt))

    # Injection sample
    i_inj = np.argmin(np.abs(t - t_inj))

    # Number of complete windows before and after
    n_before = i_inj // nwin
    n_after = (len(x) - i_inj - 1) // nwin

    start = i_inj - n_before * nwin
    stop = i_inj + n_after * nwin + 1

    x_trim = x[start:stop]
    t_trim = t[start:stop]

    i_inj_trim = i_inj - start

    return x_trim, t_trim, i_inj_trim

import numpy as np
import xarray as xr
from scipy.signal import periodogram

def windowed_periodogram(x,
                         fs,
                         window_size,
                         axis=-1,
                         time_values=None,
                         overlap=0,
                         window="boxcar",
                         **kwgs):
    """
    Compute windowed periodograms of an nD array along a specified axis.

    Parameters
    ----------
    x : array-like
        Input nD array.

    fs : float
        Sampling rate in Hz.

    window_size : int
        Window size in samples.

    axis : int, optional
        Axis along which to compute periodograms. Default is -1.

    time_values : array-like, optional
        Time values corresponding to samples along `axis`.
        If provided, window center times are computed from these values.

    overlap : int or float, optional
        Window overlap. If int, interpreted as number of samples.
        If float between 0 and 1, interpreted as fraction of `window_size`.
        Default is 0.

    window : str, tuple, or array_like, optional
        Window function passed to `scipy.signal.periodogram`.
        Examples: "boxcar", "hann", "hamming". Default is "boxcar".

    **kwgs
        Extra keyword arguments passed to `scipy.signal.periodogram`.

    Returns
    -------
    xr.DataArray
        Periodograms with dimensions (..., f, t).
    """
    x = np.asarray(x)
    axis = np.core.numeric.normalize_axis_index(axis, x.ndim)
    n = x.shape[axis]

    if window_size > n:
        raise ValueError("window_size must be <= length of selected axis")

    if isinstance(overlap, float):
        if not (0 <= overlap < 1):
            raise ValueError("float overlap must satisfy 0 <= overlap < 1")
        noverlap = int(round(overlap * window_size))
    else:
        noverlap = int(overlap)

    if not (0 <= noverlap < window_size):
        raise ValueError("overlap must satisfy 0 <= overlap < window_size")

    step = window_size - noverlap

    starts = np.arange(0, n - window_size + 1, step)
    n_windows = len(starts)

    if n_windows == 0:
        raise ValueError("No complete windows fit in the selected axis")

    if time_values is not None:
        time_values = np.asarray(time_values)
        if time_values.ndim != 1:
            raise ValueError("time_values must be 1D")
        if len(time_values) != n:
            raise ValueError("len(time_values) must equal x.shape[axis]")

    # Move analyzed axis to the end
    x_moved = np.moveaxis(x, axis, -1)

    # Build windowed array: (..., n_windows, window_size)
    x_windowed = np.stack(
        [x_moved[..., s:s + window_size] for s in starts],
        axis=-2
    )

    # Compute periodogram along last axis
    f, pxx = periodogram(
        x_windowed,
        fs=fs,
        window=window,
        axis=-1,
        **kwgs
    )

    # (..., n_windows, n_freq) -> (..., n_freq, n_windows)
    pxx = np.moveaxis(pxx, -1, -2)

    # Window center times
    if time_values is None:
        times = (starts + window_size / 2) / fs
    else:
        times = np.array([
            time_values[s:s + window_size].mean()
            for s in starts
        ])

    dims = [f"dim_{i}" for i in range(x.ndim)]
    original_dim = dims[axis]
    out_dims = dims[:axis] + dims[axis + 1:] + ["f", "t"]

    return xr.DataArray(
        pxx,
        dims=out_dims,
        coords={
            "f": f,
            "t": times,
        },
        attrs={
            "fs": fs,
            "window_size": window_size,
            "overlap": noverlap,
            "step": step,
            "window": window,
            "analyzed_axis": original_dim,
        },
    )

# def windowed_periodogram(x,
#                          fs,
#                          window_size,
#                          axis=-1,
#                          time_values=None,
#                          **kwgs
#                         ):
#     """
#     Compute windowed periodograms of an nD array along a specified axis.

#     Parameters
#     ----------
#     x : array-like
#         Input nD array.

#     fs : float
#         Sampling rate (Hz).

#     window_size : int
#         Window size in samples.

#     axis : int, optional
#         Axis along which to compute the periodograms.
#         Default is -1.

#     time_values : array-like, optional
#         Time values corresponding to samples along `axis`.
#         If provided, window center times are computed from these values.
#         Length must equal x.shape[axis].
        
#     kwgs : keyword arguments passed to scipy.signal.periodogram

#     Returns
#     -------
#     xr.DataArray
#         Periodograms with dimensions:
#         (..., f, t)
#     """
#     x = np.asarray(x)

#     axis = np.core.numeric.normalize_axis_index(axis, x.ndim)
#     n = x.shape[axis]

#     if window_size > n:
#         raise ValueError("window_size must be <= length of selected axis")

#     if time_values is not None:
#         time_values = np.asarray(time_values)
#         if time_values.ndim != 1:
#             raise ValueError("time_values must be 1D")
#         if len(time_values) != n:
#             raise ValueError(
#                 "len(time_values) must equal x.shape[axis]"
#             )

#     # Number of full non-overlapping windows
#     n_windows = n // window_size

#     # Truncate to full windows only
#     trimmed_length = n_windows * window_size

#     # Move target axis to the end
#     x_moved = np.moveaxis(x, axis, -1)

#     # Keep only complete windows
#     x_moved = x_moved[..., :trimmed_length]

#     # Reshape into (..., n_windows, window_size)
#     new_shape = x_moved.shape[:-1] + (n_windows, window_size)
#     x_windowed = x_moved.reshape(new_shape)

#     # Compute periodogram along the last axis
#     f, pxx = periodogram(x_windowed, fs=fs, axis=-1, **kwgs)

#     # (..., n_windows, n_freq) -> (..., n_freq, n_windows)
#     pxx = np.moveaxis(pxx, -1, -2)

#     # Window center times
#     if time_values is None:
#         times = (
#             np.arange(n_windows) * window_size
#             + window_size / 2
#         ) / fs
#     else:
#         time_values = time_values[:trimmed_length]
#         t_windowed = time_values.reshape(n_windows, window_size)
#         times = t_windowed.mean(axis=1)

#     # Build dimension names
#     dims = [f"dim_{i}" for i in range(x.ndim)]
#     original_dim = dims[axis]

#     out_dims = dims[:axis] + dims[axis + 1:] + ["f", "t"]

#     coords = {
#         "f": f,
#         "t": times,
#     }

#     return xr.DataArray(
#         pxx,
#         dims=out_dims,
#         coords=coords,
#         attrs={
#             "fs": fs,
#             "window_size": window_size,
#             "analyzed_axis": original_dim,
#         },
#     )