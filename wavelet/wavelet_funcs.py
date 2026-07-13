import numpy as np
import matplotlib.pyplot as plt
import pywt
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
import numpy as np
import matlab
from scipy.special import gamma

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
    
def plot_timeseries_and_its_cwt(x, t, f1=0.01, f2=0.035, axs=None, show_smoothed=True, sigma_gb=7, **kwgs):
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

def matlab_cwt(
    eng,
    x,
    Fs,
    time_bandwidth=60,
    frequency_limits=None,
    voices_per_octave=16,
    boundary="periodic",
    return_xarray=True,
):
    """
    Compute MATLAB CWT using the Morse wavelet from Python.

    Parameters
    ----------
    eng : matlab.engine.MatlabEngine
        Active MATLAB engine.

    x : array_like
        1D signal.

    Fs : float
        Sampling frequency.

    time_bandwidth : float, optional
        Morse TimeBandwidth parameter.

    frequency_limits : tuple or None, optional
        (fmin, fmax). If None, computed with MATLAB cwtfreqbounds().

    voices_per_octave : int, optional
        Voices per octave.

    boundary : str, optional
        Boundary handling.

    return_xarray : bool, optional
        If True (default), return an xarray.Dataset.
        Otherwise return raw arrays.

    Returns
    -------
    xr.Dataset or tuple
    """
    x = np.asarray(x, dtype=float).ravel()

    eng.workspace["x"] = matlab.double(x.reshape(-1, 1).tolist())
    eng.workspace["Fs"] = float(Fs)
    eng.workspace["time_bandwidth"] = float(time_bandwidth)
    eng.workspace["voices_per_octave"] = float(voices_per_octave)
    eng.workspace["boundary"] = boundary
    eng.workspace["use_auto_freq_limits"] = frequency_limits is None

    if frequency_limits is not None:
        eng.workspace["fmin"] = float(frequency_limits[0])
        eng.workspace["fmax"] = float(frequency_limits[1])

    eng.eval(
        """
        if use_auto_freq_limits
            [fmin, fmax] = cwtfreqbounds( ...
                numel(x), Fs, ...
                Wavelet="Morse", ...
                TimeBandwidth=time_bandwidth, ...
                VoicesPerOctave=voices_per_octave);
        end

        fb = cwtfilterbank( ...
            SignalLength=numel(x), ...
            SamplingFrequency=Fs, ...
            Wavelet="Morse", ...
            TimeBandwidth=time_bandwidth, ...
            FrequencyLimits=[fmin, fmax], ...
            VoicesPerOctave=voices_per_octave, ...
            Boundary=boundary);

        [cfs, f, coi] = cwt(x, FilterBank=fb);

        amp = abs(cfs);
        scales = fb.Scales;
        """,
        nargout=0,
    )

    wt_amp = np.asarray(eng.workspace["amp"])
    wt_coefs = np.asarray(eng.workspace["cfs"])
    freq = np.asarray(eng.workspace["f"]).squeeze()
    coi = np.asarray(eng.workspace["coi"]).squeeze()
    scales = np.asarray(eng.workspace["scales"]).squeeze()
    fmin = float(eng.workspace["fmin"])
    fmax = float(eng.workspace["fmax"])

    if not return_xarray:
        return wt_amp, freq, coi, scales, fmin, fmax

    t = np.arange(len(x)) / Fs

    ds = xr.Dataset(
        data_vars=dict(
            wt_coefs=(("f", "t"), wt_coefs),
            wt_amp=(("f", "t"), wt_amp),
            coi=("t", coi),
            signal=("t", x),
        ),
        coords=dict(
            f=freq,
            t=t,
            scale=("f", scales),
        ),
        attrs=dict(
            Fs=Fs,
            wavelet="Morse",
            time_bandwidth=time_bandwidth,
            voices_per_octave=voices_per_octave,
            boundary=boundary,
            frequency_min=fmin,
            frequency_max=fmax,
            matlab_engine_version=eng.version()
        ),
    )

    return ds

def admissibility_constant_gmw_matlab(gam=3, timebandwidth=60):

    """
    Source:
    https://se.mathworks.com/help/wavelet/ref/cwtfilterbank.scalespectrum.html#:~:text=cPsi%20%3D%20anorm%5E2/(2*ga).*(1/2)%5E(2*(be/ga)%2D1)*gamma(2*be/ga)%3B

    """
    beta = timebandwidth / gam

    anorm = 2*np.exp(beta/gam*(1+(np.log(gam)-np.log(beta))));
    cPsi = anorm**2 / (2*gam)*(1/2)**(2*(beta/gam)-1)*gamma(2*beta/gam);
    
    return float(cPsi)        

def coi_around_injection(t, coi, t_inject):
    """
    Construct COI around an injection time by mirroring the pre-injection COI
    and appending the post-injection COI.

    Parameters
    ----------
    t : array_like
        Time array.

    coi : array_like
        COI values corresponding to `t`.

    t_inject : float
        Injection time.

    Returns
    -------
    t_coi_injection : ndarray
        Time values corresponding to `coi_injection`.

    coi_injection : ndarray
        COI values arranged around injection.
    """
    t = np.asarray(t)
    coi = np.asarray(coi)

    if t.shape != coi.shape:
        raise ValueError("t and coi must have the same shape")

    # Left side: all samples before injection
    mask_left = t < t_inject
    coi_left = coi[mask_left][::-1]

    # Right side: from beginning up to the middle of t
    coi_right = coi[t < np.mean(t)]
    
    coi_injection = np.concatenate([coi_left, coi_right])

    # Time axis starting from original beginning, same length as constructed COI
    t_coi_injection = t[:len(coi_injection)]

    return t_coi_injection, coi_injection


def band_significance(wt_powL2, noise_band, signal_band, threshold_k=3, ignorena=True):
    """
    Estimate whether wavelet power integrated over a frequency band exceeds
    the expected noise level.

    Parameters
    ----------
    wt_powL2 : xarray.DataArray
        Wavelet power spectrum with dimensions ``('f', 't')``, where ``f`` is
        frequency and ``t`` is time. Should be (proportional to) L2-normalized
        CWT power because thresholding assumes flat noise floor.
        
    noise_band : tuple of float
        Lower and upper frequency bounds ``(f_min, f_max)`` defining the
        frequency band used to estimate the mean background noise power.
    
    signal_band : tuple of float
        Lower and upper frequency bounds ``(f_min, f_max)`` over which the
        wavelet power is integrated.
    
    threshold_k : float, default=3
        Scaling factor applied to the estimated mean noise power when computing
        the expected noise power. (Currently not used in the implementation.)
    
    ignorena : bool, default True
        Whether or not to ignore NaNs. If True, a time point is excluded if any
        of ``wt_powL2`` at this time is NaN (e.g. if cone of influence is masked out).

    Returns
    -------
    signif : xarray.DataArray
        Boolean array indexed by time indicating whether the integrated signal
        band power exceeds the estimated noise power. Time points with NaN band
        power are excluded.
    band_power_rel : xarray.DataArray
        Integrated signal-band power normalized by the expected noise power.
    noise_power : float
        Estimated integrated noise power over the signal bandwidth.
    noise_mean : float
        Mean wavelet power within the noise band.
        
    Notes
    -----
    
        - The mean wavelet power within a noise band is used to estimate
    the background noise power. The noise power integral is calculated
    as the mean wavelet noise power multiplied by `threshold_k` and
    the `signal_band` width. The wavelet power is then integrated over the
    signal band and normalized by the expected noise power integral. Time 
    points are classified as significant when the normalized band power exceeds 1.
    
        - Assumes that wavelet power `wt_powL2` is sorted along decreasing 
        frequency values.
        
    """
    noise_mean = wt_powL2[::-1].sel(f=slice(*noise_band)).mean(('f', 't')).item()
    noise_power = noise_mean * (signal_band[1] - signal_band[0])

    if ignorena:
        band_power = wt_powL2[::-1].sel(f=slice(*signal_band)).integrate('f')
    else:
        band_power = wt_powL2[::-1].sel(f=slice(*signal_band)).fillna(0).integrate('f') # Replace Nan to 0 and summ them up during integration
        
    band_power_rel = band_power / noise_power
    signif = band_power_rel > threshold_k
    signif = signif[~band_power_rel.isnull()]
    
    return signif, band_power_rel, noise_power, noise_mean 