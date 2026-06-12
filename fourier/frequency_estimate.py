import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram, find_peaks
import xarray as xr
from scipy.stats import chi2

def scan_periodogram(
                    signal,
                    fs,
                    max_drop,
                    signal_band = (0.02, 0.05),
                    detrend='linear',
                    
                ):
    """
    Localize periodogram peak while iteratively trimming the 
    signal. Return peak coordinates for all k trimmed samples.

    Parameters
    ----------
        signal : array-like
            Input 1D signal
        fs : float
            Sampling frequency (Hz)
        max_drop : int
            Max number of samples to trim
        signal_band : tuble with two floats
            Frequency band to search for the peak
        detrend : str, default 'linear'
            Passed to `scipy.signal.periodogram`. See docs for 
            `scipy.signal.periodogram` for details

    Returns
    -------
        frequency : 1d array
        periodogram values : 1d array
            
    """

    signal = np.asarray(signal)
    n = len(signal)
    
    freqs = []
    vals = []
    
    for k in range(max_drop):
        current_signal = signal[:n - k]

        f, Pxx = periodogram(current_signal, fs=fs, detrend=detrend)

        # --- Peak band ---
        band_mask = (f >= signal_band[0]) & (f <= signal_band[1])
        if not np.any(band_mask):
            continue

        f_band = f[band_mask]
        P_band = Pxx[band_mask]

        peak_idx_band = np.argmax(P_band)
        
        # Map back to full index
        full_indices = np.where(band_mask)[0]
        peak_idx_full = full_indices[peak_idx_band]
        

        freqs.append(f[peak_idx_full])
        vals.append(Pxx[peak_idx_full])

    return freqs, vals


def max_peak_periodogram(
                    signal,
                    fs,
                    period,
                    signal_band = (0.02, 0.05),
                    noise_band = (0.1, 0.9),
                    noise_k=10,
                    detrend='linear',
                    
                ):
    """
    Iteratively trims the signal from the end to make the spectral peak
    (within `signal_band`) as tall as possible Evaluates its
    significance relative to a noise floor.

    Parameters
    ----------
        signal : array-like
            Input 1D signal
        fs : float
            Sampling frequency (Hz)
        period : float
            Approximate period (s) of signal oscillation
        signal_band : tuble with two floats
            Frequency band to search for the peak
        noise_band : tuble with two floats
            Frequency band used to estimate noise level by averaging
            power within this band
        noise_k : float, default 10.0
            Peak is considered significant if its height exceedes `noise_k*noise`
        detrend : str, default 'linear'
            Passed to `scipy.signal.periodogram`. See docs for 
            `scipy.signal.periodogram` for details

    Returns
    -------
        best_result : dict
            {
                "frequencies": f,
                "power": Pxx,
                "trimmed_signal": best_signal,
                "peak_index": idx,
                "symmetry_score": How similar are the two neighbours of the peak,
                "num_trimmed": number of samples trimmed,
                "peak_height": peak_val,
                "noise_level": noise_level,
                "significant": bool
            }
            
    """

    signal = np.asarray(signal)
    n = len(signal)

    best_height = 0
    best_result = None
    
    n_drop_max = int(fs * period)

    
    # Iterate by trimming last samples
    for k in range(n_drop_max):
        current_signal = signal[:n - k]

        f, Pxx = periodogram(current_signal, fs=fs, detrend=detrend)

        # --- Peak band ---
        band_mask = (f >= signal_band[0]) & (f <= signal_band[1])
        if not np.any(band_mask):
            continue

        f_band = f[band_mask]
        P_band = Pxx[band_mask]

        peak_idx_band = np.argmax(P_band)
        
        # Map back to full index
        full_indices = np.where(band_mask)[0]
        peak_idx_full = full_indices[peak_idx_band]
        peak_height = Pxx[peak_idx_full]

        if peak_height > best_height:
            best_height = peak_height

            best_result = {
                "frequencies": f,
                "power": Pxx,
                "trimmed_signal": current_signal,
                "peak_index": peak_idx_full,
                "peak_freq": f[peak_idx_full],
                "peak_height": peak_height,
                "num_trimmed": k,
            }
            
    # --- Noise estimation ---

    noise_mask = (f >= noise_band[0]) & (f <= noise_band[1])

    # Exclude peak band from noise estimate (important!)
    noise_mask &= ~band_mask

    if np.any(noise_mask):
        noise_level = np.mean(Pxx[noise_mask])
    else:
        noise_level = np.nan
        print("No values in the noise band. Cannot estimate noise level.")
    
    if best_result is not None:
        best_result["noise_level"] = noise_level
        best_result['noise_k'] = noise_k
        best_result["significant"] = peak_height >= noise_k * noise_level
        best_result['signal_band_f1'] = signal_band[0]
        best_result['signal_band_f2'] = signal_band[1]
        best_result['noise_band_f1'] = noise_band[0]
        best_result['noise_band_f2'] = noise_band[1]
    
    return best_result

def symmetric_peak_periodogram(
                    signal,
                    fs,
                    period,
                    signal_band = (0.02, 0.05),
                    noise_band = (0.1, 0.9),
                    noise_k=10,
                    detrend='linear'
                ):
    """
    Iteratively trims the signal from the end to make the spectral peak
    (within `signal_band`) as symmetric as possible. Evaluates its
    significance relative to a noise floor.

    Parameters
    ----------
        signal : array-like
            Input 1D signal
        fs : float
            Sampling frequency (Hz)
        period : float
            Approximate period (s) of signal oscillation
        signal_band : tuble with two floats
            Frequency band to search for the peak
        noise_band : tuble with two floats
            Frequency band used to estimate noise level by averaging
            power within this band
        noise_k : float, default 10.0
            Peak is considered significant if its height exceedes `noise_k*noise`
        detrend : str, default 'linear'
            Passed to `scipy.signal.periodogram`. See docs for 
            `scipy.signal.periodogram` for details

    Returns
    -------
        best_result : dict
            {
                "frequencies": f,
                "power": Pxx,
                "trimmed_signal": best_signal,
                "peak_index": idx,
                "symmetry_score": How similar are the two neighbours of the peak,
                "num_trimmed": number of samples trimmed,
                "peak_height": peak_val,
                "noise_level": noise_level, computed as the mean PSD in the noise band,
                "significant": bool
            }
            
    """

    signal = np.asarray(signal)
    n = len(signal)

    best_score = 0
    best_result = None
    
    n_drop_max = int(fs * period)

    # Iterate by trimming last samples
    for k in range(n_drop_max):
        current_signal = signal[:n - k]

        f, Pxx = periodogram(current_signal, fs=fs, detrend=detrend)

        # --- Peak band ---
        band_mask = (f >= signal_band[0]) & (f <= signal_band[1])
        if not np.any(band_mask):
            continue

        f_band = f[band_mask]
        P_band = Pxx[band_mask]

        peak_idx_band = np.argmax(P_band)

        # Need neighbors on both sides
        if peak_idx_band == 0 or peak_idx_band == len(P_band) - 1:
            continue

        left = P_band[peak_idx_band - 1]
        right = P_band[peak_idx_band + 1]

        score =  P_band[peak_idx_band] - np.mean([left, right])

        if score > best_score:
            best_score = score

            # Map back to full index
            full_indices = np.where(band_mask)[0]
            peak_idx_full = full_indices[peak_idx_band]
            peak_val = Pxx[peak_idx_full]

            best_result = {
                "frequencies": f,
                "power": Pxx,
                "trimmed_signal": current_signal,
                "peak_index": peak_idx_full,
                "peak_freq": f[peak_idx_full],
                "peak_height": peak_val,
                "symmetry_score": score,
                "num_trimmed": k,
            }
            
    # --- Noise estimation ---

    noise_mask = (f >= noise_band[0]) & (f <= noise_band[1])

    # Exclude peak band from noise estimate (important!)
    noise_mask &= ~band_mask

    if np.any(noise_mask):
        noise_level = np.mean(Pxx[noise_mask])
    else:
        noise_level = np.nan
        print("No values in the noise band. Cannot estimate noise level.")
    
    if best_result is not None:
        best_result["noise_level"] = noise_level
        best_result['noise_k'] = noise_k
        best_result["significant"] = peak_val >= noise_k * noise_level
        best_result['signal_band_f1'] = signal_band[0]
        best_result['signal_band_f2'] = signal_band[1]
        best_result['noise_band_f1'] = noise_band[0]
        best_result['noise_band_f2'] = noise_band[1]
    
    return best_result


def plot_results(res, ax=None, label=None, stepplot=False):
    """Plot spectrum, detected peak, and noise level. Labels
    the peak with `*` in the legend if the peak height is significant
    
    Parameters
    ----------
        res : dict
            Output of `symmetric_peak_periodogram` function.
            Contains power values, frequency values, peak freq.
            and height, noise level, significance, etc.
        ax : axes, optional
        label : str, optional
        stepplot : bool, default False
            Whether to plot spectrum using pyplot.step
            with parameter where='mid'.

    """
    
    return_ax = False
    if ax is None:
        _,ax = plt.subplots()
        return_ax = True
    if stepplot:
        ax.step(res['frequencies'], res['power'], color='k', where='mid')
    else:
        ax.plot(res['frequencies'], res['power'], 'k.-')
    
    ax.plot(res['peak_freq'], res['peak_height'], marker='o', color='r', label="*" if res['significant'] else "n.s.")
    ax.axhline(res['noise_level'], 
               color='k', label='noise', ls='--', alpha=.3)
    
    kk = res['noise_k']
    ax.axhline(kk*res['noise_level'], 
               color='r', label=f'{kk}*noise', ls='--', alpha=.3)    
    ax.legend()
    ax.set(xlabel="Frequency (Hz)", ylabel="Spectral power")
    return ax

def detect_periodogram_peaks(
    freq,
    power,
    noise_band,
    signal_band,
    n_averages=1,
    p_global_max=None,
    prominence=None,
    distance=None,
    return_xarray=True,
):
    """
    Detect significant peaks in a periodogram.

    Parameters
    ----------
    freq : array-like
        Frequency array.
    power : array-like
        Periodogram values.
    n_averages : float
        Effective number of spectrograms averaged to get
        `power`.
    noise_band : tuple(float, float)
        Frequency interval used to estimate background noise.
        Example: (0.30, 0.45)
    signal_band : tuple(float, float)
        Frequency interval in which peaks are searched.
        Example: (0.01, 0.20)
    p_global_max : float or None
        If given, only peaks with p_global <= p_global_max
        are retained.
    prominence : float or None
        Passed to scipy.signal.find_peaks.
    distance : int or None
        Passed to scipy.signal.find_peaks.
    return_xarray : bool
        If True return xarray.Dataset,
        otherwise return dict.

    Returns
    -------
    xarray.Dataset or dict
    """

    freq = np.asarray(freq)
    power = np.asarray(power)

    # ----------------------------
    # Estimate background level
    # ----------------------------
    noise_mask = (
        (freq >= noise_band[0]) &
        (freq <= noise_band[1])
    )

    if noise_mask.sum() == 0:
        raise ValueError("noise_band contains no frequencies")

    P0 = power[noise_mask].mean()

    # ----------------------------
    # Search region
    # ----------------------------
    signal_mask = (
        (freq >= signal_band[0]) &
        (freq <= signal_band[1])
    )

    signal_indices = np.where(signal_mask)[0]

    if len(signal_indices) == 0:
        raise ValueError("signal_band contains no frequencies")

    local_power = power[signal_mask]

    peaks_local, props = find_peaks(
        local_power,
        prominence=prominence,
        distance=distance,
    )

    peak_idx = signal_indices[peaks_local]

    # Number of tested frequencies
    N = signal_mask.sum()

    peak_freq = freq[peak_idx]
    peak_power = power[peak_idx]

    # ----------------------------
    # Statistics - compute false alarm probability
    # ----------------------------
    z = peak_power / P0

    M = int(n_averages)

    if M < 1:
        raise ValueError("n_averages must be >= 1")

    if M == 1:
        # raw periodogram
        p_single = np.exp(-z)
    else:
        # Welch / averaged periodogram
        p_single = chi2.sf(
            2 * M * z,
            df=2 * M,
        )

    p_global = 1.0 - (1.0 - p_single) ** N

    # ----------------------------
    # Optional significance filter
    # ----------------------------
    if p_global_max is not None:
        keep = p_global <= p_global_max

        peak_idx = peak_idx[keep]
        peak_freq = peak_freq[keep]
        peak_power = peak_power[keep]
        z = z[keep]
        p_single = p_single[keep]
        p_global = p_global[keep]

    # ----------------------------
    # Output
    # ----------------------------
    if return_xarray:

        ds = xr.Dataset(
            data_vars=dict(
                frequency=("peak", peak_freq),
                power=("peak", peak_power),
                Z=("peak", z),
                p_single=("peak", p_single),
                p_global=("peak", p_global),
                index=("peak", peak_idx),
            ),
            attrs=dict(
                P0=float(P0),
                noise_band=noise_band,
                signal_band=signal_band,
                N_tested=int(N),
                N_averaged_spectra= int(M),
            ),
        )

        return ds

    return {
        "indices": peak_idx,
        "frequencies": peak_freq,
        "power": peak_power,
        "P0": P0,
        "Z": z,
        "p_single": p_single,
        "p_global": p_global,
        "N_tested": N,
        "num_averaged_spectra": M,
    }


def detect_significant_bins(
    freq,
    power,
    noise_band,
    signal_band,
    n_averages=1,
    p_single_max=0.01,
    return_xarray=True,
):
    """
    Detect all frequency bins significantly above background.

    Parameters
    ----------
    p_single_max : float
        Maximum single-bin false alarm probability.

    Returns
    -------
    xarray.Dataset or dict
    """

    freq = np.asarray(freq)
    power = np.asarray(power)

    M = int(n_averages)

    if M < 1:
        raise ValueError("n_averages must be >= 1")

    noise_mask = (
        (freq >= noise_band[0]) &
        (freq <= noise_band[1])
    )

    signal_mask = (
        (freq >= signal_band[0]) &
        (freq <= signal_band[1])
    )

    P0 = power[noise_mask].mean()

    idx = np.where(signal_mask)[0]

    z = power[idx] / P0

    if M == 1:
        p_single = np.exp(-z)
    else:
        p_single = chi2.sf(
            2 * M * z,
            df=2 * M,
        )

    keep = p_single <= p_single_max

    bin_idx = idx[keep]

    if return_xarray:

        return xr.Dataset(
            data_vars=dict(
                frequency=("bin", freq[bin_idx]),
                power=("bin", power[bin_idx]),
                Z=("bin", power[bin_idx] / P0),
                p_single=("bin", p_single[keep]),
                index=("bin", bin_idx),
            ),
            attrs=dict(
                P0=float(P0),
                noise_band=noise_band,
                signal_band=signal_band,
                p_single_max=float(p_single_max),
                N_averaged_spectra=M,
                freq_step = np.diff(freq)[0]
            ),
        )

    return {
        "indices": bin_idx,
        "frequencies": freq[bin_idx],
        "power": power[bin_idx],
        "Z": power[bin_idx] / P0,
        "p_single": p_single[keep],
        "P0": P0,
        "N_averaged_spectra":M,
        "freq_step": np.diff(freq)[0]
    }