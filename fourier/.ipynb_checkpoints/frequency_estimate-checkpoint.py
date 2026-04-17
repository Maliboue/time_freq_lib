import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram


def symmetric_peak_periodogram(
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
                "noise_level": noise_level,
                "significant": bool
            }
            
    """

    signal = np.asarray(signal)
    n = len(signal)

    best_score = np.inf
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

        score = abs(left - right)

        if score < best_score:
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
                "symmetry_score": score,
                "num_trimmed": k,
                "peak_height": peak_val,
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


def plot_results(res, ax=None, label=None):
    """Plot spectrum, detected peak, and noise level. Labels
    the peak with `*` in the legend if the peak height is significant
    
    Parameters
    ----------
        res : dict
            Output of `symmetric_peak_periodogram` function.
            Contains power values, frequency values, peak freq.
            and height, noise level, significance, etc.
        ax : axes, optional

    """
    
    return_ax = False
    if ax is None:
        _,ax = plt.subplots()
        return_ax = True
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

# def symmetric_peak_periodogram(
#                     signal,
#                     fs,
#                     period,
#                     fmin=0.02,
#                     fmax=0.04
#                 ):
#     """
#     Iteratively trims the signal from the end to make the spectral peak
#     (within [fmin, fmax]) as symmetric as possible.

#     Parameters
#     ----------
#         signal : array-like
#             Input 1D signal
#         fs : float
#             Sampling frequency (Hz)
#         period : float
#             Approximate period (s) of sigmal oscillation.
#         fmin, fmax : float
#             Frequency band to search for the peak


#     Returns
#     -------
#         best_result : dict
#             {
#                 "frequencies": f,
#                 "power": Pxx,
#                 "trimmed_signal": best_signal,
#                 "peak_index": idx,
#                 "symmetry_score": score,
#                 "num_trimmed": number of samples trimmed
#             }

#     Note
#     ----
#         - The maximum number of samples to be dropped from 
#         the end of the signal is estimated as int(period * fs).
        
#         - See an explanation in section B in 
#         Kutuzov et al. 2025 (DOI: 10.1063/5.0225333).
#     """

#     signal = np.asarray(signal)
#     n = len(signal)

#     best_score = np.inf
#     best_result = None
    
#     n_drop_max = int(fs * period)

#     # Iterate by trimming last samples
#     for k in range(n_drop_max):
#         current_signal = signal[:n - k]

#         f, Pxx = periodogram(current_signal, fs=fs)

#         # Restrict to frequency band
#         band_mask = (f >= fmin) & (f <= fmax)
#         if not np.any(band_mask):
#             continue

#         f_band = f[band_mask]
#         P_band = Pxx[band_mask]

#         # Find peak in band
#         peak_idx_band = np.argmax(P_band)

#         # Need neighbors on both sides
#         if peak_idx_band == 0 or peak_idx_band == len(P_band) - 1:
#             continue

#         # Neighbor values
#         left = P_band[peak_idx_band - 1]
#         right = P_band[peak_idx_band + 1]

#         # Symmetry criterion: absolute difference
#         score = abs(left - right)

#         # Optional normalization (uncomment if desired)
#         # peak_val = P_band[peak_idx_band]
#         # score = abs(left - right) / peak_val

#         if score < best_score:
#             best_score = score

#             # Map back to full index
#             full_indices = np.where(band_mask)[0]
#             peak_idx_full = full_indices[peak_idx_band]

#             best_result = {
#                 "frequencies": f,
#                 "power": Pxx,
#                 "trimmed_signal": current_signal,
#                 "peak_index": peak_idx_full,
#                 "symmetry_score": score,
#                 "num_trimmed": k
#             }
    
    

#     return best_result
