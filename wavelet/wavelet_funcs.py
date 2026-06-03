import numpy as np
import matplotlib.pyplot as plt
import pywt
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter

import numpy as np

def morlet_coi_ssq(scales, dt=1.0):
    """
    Cone-of-influence half-width for ssqueezepy Morlet wavelet.

    Parameters
    ----------
    scales : array_like
        CWT scales.

    dt : float
        Sampling interval.

    Returns
    -------
    coi_half_width : ndarray
        COI half-width in time units.

    Note
    ----
    This assumes that the wavelet's envelope is exp(-t^2/2) (ssqueezepy, Torrece & Compo 1998).
    
    """
    scales = np.asarray(scales)
    return np.sqrt(2) * scales * dt

def intersection_between_coi_and_if_cmor(coi, ifreq, t, bw, cf, at_the_end=False):
    
    """Find intersection between cone of influence and instantaneous
    frequency for complex Morlet wavelet.
    
    Parameters:
    
     - coi : 1d array (N,)
         Cone of influence.
     - ifreq : 1d array (K,)
         Instantaneous frequency.
     - t : 1d array (K,)
         Time vector.
     - bw, cf : float, float
         Bandwidth and central frequency parameters of the complex 
         Morlet wavelet (see definition in PyWavelets docs).    
     - at_the_end : bool, default False
         If True, computes intersection coordinates at the end
         of the time axis. If False, computes the coordinates at
         the start of the time axis.
         
    Returns:
    
     - time and frequency at which COI and IF intersect
     
     
    Note:
    
     - Does no interpolation.
    
    """
    
    coi = np.asarray(coi)
    ifreq = np.asarray(ifreq)
    t = np.asarray(t)
    
    if at_the_end:
        ifreq = ifreq[::-1]
    
    t0, t1 = np.min(coi), np.max(coi)
    sele = (t>t0)&(t<t1)

    ifreq_sele = ifreq[sele]
    t_sele = t[sele]
    coi_freq_fine = np.sqrt(bw)*cf/t_sele

    intersection_idx = np.argmin(abs(coi_freq_fine - ifreq_sele))
    
    freq_intersection = ifreq_sele[intersection_idx]
    
    if at_the_end:
        return (t[t.size - (intersection_idx + sum(t<=t0))], 
                freq_intersection)
    else:
        return (t_sele[intersection_idx], 
                freq_intersection)
    


def wavelet_ridge(W, freqs, times, return_xarray=True):
    """
    Compute ridge from complex wavelet coefficients by taking
    maximum along frequency axis.

    Parameters
    ----------
    W : np.ndarray (n_freq, n_time)
        Complex wavelet coefficients.
    freqs : np.ndarray
        Frequency vector of length n_freq.
    times : np.ndarray
        Time vector of length n_time.
    return_xarray : bool, default True
        If True, return an xarray.Dataset.

    Returns
    -------
    ds : xarray.Dataset with
        ridge_freq : np.ndarray (n_time,)
        ridge_amp : np.ndarray (n_time,)
            Absolute value of ridge coefficients.
        ridge_phase : np.ndarray (n_time,)
            Phase angle of ridge coefficients.
            
    tuple of these variables if return_xarray is False.
    
    """

    W = np.asarray(W)
    freqs = np.asarray(freqs)
    times = np.asarray(times)
    
    W = np.asarray(W)
    if W.ndim != 2:
        raise ValueError("W must be 2D (n_freq, n_time)")

    # Amplitude
    amp = np.abs(W)

    # Ridge: index of max amplitude per time
    ridge_freq_idx = np.argmax(amp, axis=0)
    ridge_freq = freqs[ridge_freq_idx]
    ridge_coefs = W[ridge_freq_idx, np.arange(len(times))]
    ridge_amp = np.abs(ridge_coefs)
    ridge_phase = np.angle(ridge_coefs)
    
    if return_xarray:
        return xr.Dataset(
                    data_vars={
                        "frequency": ("t", ridge_freq),
                        "amplitude": ("t", ridge_amp),
                        "phase": ("t", ridge_phase),
                    },
                    coords={
                        "t": times,
                    },
                )
    else:
        return (ridge_freq, 
                ridge_amp, 
                ridge_phase
               )

def cwt_with_coi(data, 
                 time, 
                 freq_min, 
                 freq_max, 
                 bandwidth=1.5, 
                 central_freq=1.0, 
                 nscales=128,
                 return_xarray=True,
                 padding_fraction=0.1, 
                 pad_mode='symmetric',
                 **kwargs):
    """Compute continuous wavelet transform (CWT) of data, and the
    cone of influence (COI). Uses Complex Morlet wavelet.
    
    Parameters:
    
        - data : 1d array
            Data values.
        
        - time : 1d array
            Data time values.
        
        - freq_min, freq_max : float, float
            lower and higher frequency, between which to compute
            CWT.
            
        - bandwidth, central_freq : float, float
            Parameters of the complex morlet wavelet. Converted
            to string 'cmor{bandwidth}-{central_freq}' and passed to
            `pywavelets.cwt`. In Pywavelets documentation `bandwidth`
            and `central_freq` are designated as B and C, respectively.
            
        - nscales : int
            Number of scales (frequencies) in the CWT.
        
        - padding_fraction : float, default 0.1
            Fraction of the length of the data to pad. This determines
            how much padding to add to each side of the input data.
        
        - pad_mode : str, default 'symmetric'
            Symmetric because this hides the edge artifacts, easing visualization
            of coefficients far from the edges.
        
        - return_xarray : bool, default True
            Whether to return the result in xarray.Dataset or not. If False,
            returns a tuple (coefs, freqs, coi)
        
        - kwargs : dict
            Keyword arguments passed to `pywavelets.cwt`. E.g., if the spectrogram 
            contains zipper artifacts, increase `precision` argument, which
            is =12 by default.
            
    Returns:
        
        - xarray.Dataset with coefs, freqs, coi
        
        - tuple (coefs, freqs, coi) if return_xarray==False
        
    Notes:
    
        - COI are computed as sqrt(2)*s, where s is the scale. This is the distance
        (in time) from the center of the wavelet, where its power decreases by
        exp(-2).
        
        - Complex Morlet wavelet, as defined in PyWavelets, has two parameters: the
        bandwidth, which is the variance of the modulating Gaussian, and the central 
        frequency, which is the frequency (Hz) of the modulated compex exponent.
        
        - Scales are computed from central_freq/freq_max to
        central_freq/freq_min using numpy.geomspace and divided by dt.
    """
    
        # Wavelet transform setup
    wavelet = f'cmor{bandwidth}-{central_freq}'
    dt = np.diff(time).mean()
        
    # Calculate the padding size
    pad_len = int(len(data) * padding_fraction)
    
    # Pad the data and time arrays
    padded_data = np.pad(data, (pad_len, pad_len), mode=pad_mode)
    padded_time = np.linspace(time[0] - pad_len * dt,
                              time[-1] + pad_len * dt,
                              len(padded_data))

    scales = np.geomspace(central_freq/freq_max, central_freq/freq_min, nscales) * (1/dt) # Central freq is in units of wavelet x-values, not time.
    
    coi = np.sqrt(bandwidth) * scales * dt 
    
    wt_coefs, wt_freqs = pywt.cwt(padded_data, scales=scales, wavelet=wavelet, sampling_period=dt, **kwargs)
    wt_coefs = wt_coefs[:, pad_len:len(padded_data)-pad_len] # remove padding
    
    if not return_xarray:
        return wt_coefs, wt_freqs, coi
    else:
        return xr.Dataset(
            data_vars=dict(
                wt_coefs=(('f', 't'), wt_coefs),
                wt_amp = (('f', 't'), abs(wt_coefs)),
                coi=('f', coi),
                scales=('f', scales)
            ),
            coords=dict(
                f=wt_freqs, 
                t=time
            ),
            attrs=dict(wavelet=wavelet, 
                       cwt_computed_by=f"{pywt.__name__}_v{pywt.__version__}",
                       padding_fraction=padding_fraction,
                       pad_mode=pad_mode
                      ),
        )
    
    
def plot_cwt(coefs, 
             freqs, 
             times, 
             coi=None, 
             ax=None, 
             add_colorbar=False, 
             cbar_kwg=dict(cbar_frac=0.1, 
                           cbar_width=0.01,
                           cbar_pad=0.01, 
                           label=''), 
             **kwargs):
    """
    Plot a Continuous Wavelet Transform (CWT) as a time-frequency heatmap.

    Parameters
    ----------
    coefs : 2D array-like
        Wavelet coefficients to plot. Shape should be (len(freqs), len(times)).
    freqs : 1D array-like
        Frequencies corresponding to the rows of `coefs`.
    times : 1D array-like
        Time points corresponding to the columns of `coefs`.
    coi : 1D array-like or None, optional
        Cone of influence (COI) curve to overlay on the plot. If provided, a dashed
        white line is drawn showing the COI boundaries.
    ax : matplotlib.axes.Axes or None, optional
        Axes object on which to draw the plot. If None, a new figure and axes
        are created.
    add_colorbar : bool, optional
        If True, adds a colorbar to the right of the axes.
    cbar_kwg : dict, optional
        Dictionary specifying colorbar properties:
            - 'cbar_frac' : fraction of height to offset the colorbar from bottom/top (default 0.1)
            - 'cbar_width' : width of the colorbar in figure coordinates (default 0.01)
            - 'cbar_pad' : horizontal padding between axes and colorbar (default 0.01)
            - 'label' : label for the y-axis of the colorbar (default '')
    **kwargs : additional keyword arguments
        Passed to `Axes.pcolormesh`, e.g., `cmap`, `vmin`, `vmax`.

    Returns
    -------
    tuple
        Tuple containing objects depending on input arguments:
        - If `ax` is None, the returned tuple starts with the `Figure` and `Axes` objects.
        - The `QuadMesh` object returned by `pcolormesh`.
        - If `add_colorbar` is True, the `Axes` of the colorbar is also included.

    Notes
    -----
    - The function rasterizes the pcolormesh for faster rendering of large datasets.
    - The colorbar, if added, is positioned to the right of the main axes with ticks
      and label on the right side, without shrinking the main axes.
    - The cone of influence (COI) can be optionally plotted as a dashed line.
    """
        
    returned = []
    if not ax:
        f, ax = plt.subplots(figsize=(10, 4))
        returned.append(f)
        returned.append(ax)
    
    pc = ax.pcolormesh(times, freqs, coefs, rasterized=True, **kwargs)
    
    if coi is not None:
        
        coi_left = coi if times.min()>0 else coi + times.min() # tiem.min()<0 e.g. in cases where an injection time is set to zero
        coi_right = times.max() - coi # Assumes time.max() is always > 0
        
        ax.plot(coi_left, freqs, 
                coi_right, freqs,
                color='w', ls='--', alpha=0.5)

    if add_colorbar:
        # divider = make_axes_locatable(ax)
        # cax     = divider.append_axes("right", size="3%", pad=0.07) 
        # plt.colorbar(im, cax=cax)
        
        ## Add colorbars
        bbox = ax.get_position()
        height = bbox.y1-bbox.y0
        cax = plt.gcf().add_axes([bbox.x1+cbar_kwg['cbar_pad'], 
                           bbox.y0 + cbar_kwg['cbar_frac']*height,
                           cbar_kwg['cbar_width'], 
                           height*(1-cbar_kwg['cbar_frac']*2)])  # [left, bottom, width, height] in figure coords
        cax.yaxis.tick_right()
        cax.yaxis.set_label_position("right")
        cbar = plt.colorbar(pc, cax=cax)
        cax.set_ylabel(cbar_kwg['label'])
        
        returned.append(cax)
    returned.append(pc)
    
    return tuple(returned)
    
def plot_timeseries_and_its_cwt(x, t, f1=0.01, f2=0.035, axs=None, 
                                show_smoothed=True, sigma_gb=7, **kwgs):
    """
    Plot time series `x` and its continuous wavelet transform (CWT).
    
    Parameters:
    
     - x : 1D array
     - t : 1D array
     - f1, f2 : floats
         Frequencies between which to compute CWT.
     - axs : tuple
         Axes on which to plot.
     - show_smoothed : bool, default True
         Whether to show Gaussian-smoothed `x`. Sigma of the Gaussian
         kernel is `sigma_gb`
     - sigma_gb : float, default 7
         Sigma, in samples, of the Gaussian kernel used for smoothing
         `x`. Only used if `show_smoothed`==True.
     - kwgs : dict
         Keyword arguments passed to `cwt_with_coi` function.
         
    Returns:
    
     - fig, (ax1, ax2) if axs is None
     
    """

    wt = cwt_with_coi(x, t, f1, f2, **kwgs)

    return_fig = False
    if axs is None:
        f, (ax,ax1) = plt.subplots(2,1,figsize=(12,7), sharex=True)
        plt.subplots_adjust(hspace=0.1)
        return_fig = True
    else:
        ax, ax1 = axs
        
    
    if show_smoothed:
        ax.plot(t, x, 'k', alpha=0.2, rasterized=True)
        ax.plot(t, gaussian_filter(x, sigma=sigma_gb), 'k', rasterized=True)
    else:
        ax.plot(t, x, 'k', rasterized=True)
    
    # ax.spines[['top', 'right']].set_visible(False)

    plot_cwt(abs(wt.wt_coefs), wt.f, wt.t, coi=wt.coi, ax=ax1)
    # ax1.set_yscale('log')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (Hz)')
    ax.set_ylabel('Contrast (dim. less)')
    
    ax1.set_xlim(wt.t.min(), wt.t.max())
    ax.set_xlim(wt.t.min(), wt.t.max())
    
    if return_fig:
        return f, (ax, ax1)
    
    

def scaled_wavelets_pywt(wavelet_name, scales, dt=1.0, length=4096):
    """
    Generate scaled wavelets from a PyWavelets continuous mother wavelet.

    Returns
    -------
    t : ndarray
        Common time grid.

    wavelets : ndarray
        Complex array, shape (n_scales, n_time).

    freqs : ndarray
        PyWavelets scale-to-frequency mapping.
    """
    wavelet = pywt.ContinuousWavelet(wavelet_name)

    # mother wavelet samples
    psi, x = wavelet.wavefun(length=length)

    scales = np.asarray(scales)

    # common time support large enough for largest scale
    t_min = x.min() * scales.max()
    t_max = x.max() * scales.max()
    t = np.arange(t_min, t_max, dt)

    wavelets = []

    for s in scales:
        # scaled wavelet: psi_s(t) = 1/sqrt(s) psi(t/s)
        x_query = t / s

        real = np.interp(x_query, x, np.real(psi), left=0, right=0)
        imag = np.interp(x_query, x, np.imag(psi), left=0, right=0)

        psi_s = (real + 1j * imag) / np.sqrt(s)
        wavelets.append(psi_s)

    wavelets = np.asarray(wavelets)

    freqs = pywt.scale2frequency(wavelet, scales) / dt

    return t, wavelets, freqs