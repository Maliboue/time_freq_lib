import numpy as np
from scipy.signal import periodogram

import numpy as np
from scipy.signal import periodogram

def symmetric_peak_periodogram(
                    signal,
                    fs,
                    period,
                    signal_band = (0.02, 0.05),
                    noise_band = (0.1, 0.9)
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

        f, Pxx = periodogram(current_signal, fs=fs)

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
    
    best_result["noise_level"] = noise_level
    best_result["significant"] = peak_val >= 3 * noise_level

    return best_result

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
