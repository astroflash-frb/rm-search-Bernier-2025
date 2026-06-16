"""
Utility functions for RM search pipeline, including RFI flagging, data normalization, and RM synthesis.
"""

import numpy as np
import astropy.units as u
import baseband_operations as bo
import pulsarbat as pb

# DEBUGGING
def print_mem(msg):
    import psutil, os
    rss = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    print(f"{msg}: {rss:.1f} GB")


def read_stokes(source_dir, obs_night, start_frame, number_of_frames, dm, ref_freq, n_pixels_to_avg):
    # Reshape and open the files for all frequencies (for chosen obs night) into readers
    my_readers = bo.get_chime_readers(source_dir, internal=obs_night)
    total_num_frames = my_readers[0].shape[0]

    # Get a specific amount of data (in time) from the readers (Nframes, Nfreq, Npol)
    dual_pol_signal = bo.lazy_read(start_frame, number_of_frames, my_readers)  # DualPolarizationSignal object
    print_mem("after lazy_read")
    print(f"Read in data with shape (Nframes, Nfreqs, Npols) = {dual_pol_signal.shape}.")

    # Dedisperse & get stokes
    dedisp_signal = pb.coherent_dedispersion(dual_pol_signal, dm, ref_freq=ref_freq)
    print_mem("after dedispersion object")
    stokes_signal = dedisp_signal.to_stokes()
    print_mem("after to_stokes")
    stokes_signal = stokes_signal.compute()  # FullStokesSignal object
    print_mem("after compute")

    # Get time and frequency arrays
    ntimes = stokes_signal.shape[stokes_signal.get_axis('time')]  # num frames after dedispersion
    time = np.linspace(0*u.s, stokes_signal.time_length - stokes_signal.dt, ntimes)
    freq = np.linspace(stokes_signal.min_freq, stokes_signal.max_freq, stokes_signal.nchan)

    # Downsample Stokes arrays and time array
    stokes_arr = np.array([
        stokes_signal.stokesI, 
        stokes_signal.stokesQ, 
        stokes_signal.stokesU, 
        stokes_signal.stokesV
    ])
    full_stokes = np.array([
        bo.shrink_any_2(d, [n_pixels_to_avg,1])  # d = 2d array w/ time on axis 0; [new time bins, new axis 1 bins]
        for d in stokes_arr
    ])  # shape (Nstokes, Ntimes, Nfreqs)
    time = bo.shrink_any_2(time.to(u.s)[:,None], [n_pixels_to_avg,1]).flatten()
    freq = freq.to(u.MHz).value
    final_num_frames = full_stokes.shape[1]

    # time info dict to return
    frame_info_dict = {
        'total_num_frames' : total_num_frames,
        'num_frames_after_dedispersion' : ntimes,
        'num_frames_stokes' : final_num_frames
    }

    return full_stokes, time, freq, frame_info_dict


def flag_rfi_manual(ranges, freqs):
    """
    ranges : list of tuples
        list of (start,end) frequencies to mask
    """
    channel_mask_manual = np.full(len(freqs), False)
    
    for (start,end) in ranges:
        # Check that freqs to mask are at least particlaly contained in the frequency array
        if start > freqs[-1] or end < freqs[0]:  # completely out of range
            continue

        if start == end:
            end += freqs[1] - freqs[0]

        mask = (start <= freqs) & (freqs <= end)
        channel_mask_manual |= mask  

    return channel_mask_manual
        

def get_rfi_mask(intensity_array, mean_treshhold=2, std_threshold=5, ranges=None, freqs=None):
    """
    Flag frequency channels with anomalous statistics across time, such as unusually high
    mean or standard deviation, which may indicate RFI. Returns a boolean mask.
    
    Parameters
    ----------
    intensity_array : np.ndarray
        Normalized intensity data with shape (Ntimes, Nfreqs), where each column is a frequency channel.
    mean_threshold : float
        Threshold for the absolute value of the mean. Channels with |mean| > mean_threshold are flagged.
    std_threshold : float
        Threshold for the standard deviation. Channels with std > std_threshold are flagged.
    
    Returns
    -------
    channel_mask : np.ndarray
        Boolean array of shape (Nfreqs,) where True indicates a flagged (to be masked) frequency channel.
    """

    # Mean and std per frequency channel (avg across time)
    mean_per_channel = np.nanmean(intensity_array, axis=0)
    std_per_channel = np.nanstd(intensity_array, axis=0)

    # Normalize & compare channel means
    means_norm = (mean_per_channel - np.mean(mean_per_channel)) / np.std(mean_per_channel)
    channel_mask_mean = np.abs(means_norm) > mean_treshhold

    # Normalize & compare channel stds
    stds_norm = (std_per_channel - np.mean(std_per_channel)) / np.std(std_per_channel)
    channel_mask_std = np.abs(stds_norm) > std_threshold

    # Manually mask some channels
    cond_manual = (ranges is not None) and (freqs is not None)
    if cond_manual:
        channel_mask_manual = flag_rfi_manual(ranges, freqs)
    
    # Combine masks
    if cond_manual:
        channel_mask = channel_mask_mean | channel_mask_std | channel_mask_manual
    else:
        channel_mask = channel_mask_mean | channel_mask_std

    return channel_mask


def normalize_data(data_array):
    """
    Normalize data by subtracting the mean and dividing by the standard deviation,
    computed per frequency channel (over time) for each Stokes parameter.

    Parameters
    ----------
    data_array : np.ndarray
        Array of data to normalize (shape: [Nstokes, Ntimes, Nfreqs]).

    Returns
    -------
    data_normalized : np.ndarray
        Normalized array of the same shape
    """
    
    mean_per_channel = np.nanmean(data_array, axis=1, keepdims=True)
    std_per_channel = np.nanstd(data_array, axis=1, keepdims=True)
    std_per_channel[std_per_channel==0] = 1
    
    return (data_array - mean_per_channel) / std_per_channel


def slice_around_burst(full_stokes, time, slice_width=0.1):
    """
    Slice the data around the highest SNR burst in the time series.
    
    Parameters
    ----------
    full_stokes : np.ndarray
        Array of shape (Nstokes, Ntimes, Nfreqs) containing the Stokes parameters.
    time : np.ndarray
        Array of time values corresponding to the time axis of full_stokes (shape: [Ntimes,]).
    slice_width : float
        Width of the time slice around the burst to keep (in seconds). The slice will be
        centered on the time of the highest SNR burst and will extend slice_width seconds before and after it.
    
    Returns
    -------
    new_stokes : np.ndarray
        Sliced array of shape (Nstokes, Ntimes_slice, Nfreqs) containing the data around the burst.
    time_burst : np.ndarray
        Time array corresponding to the new_stokes data (shape: [Ntimes_slice,]), centered around the burst.
    """

    # Find highest burst in timeseries 
    max_burst_t = time[np.argmax(np.nanmean(full_stokes[0], axis=1))]  # time of peak I
    tstart, tstop = max_burst_t - slice_width, max_burst_t + slice_width  # slice width in seconds
    tstart_ind, tstop_ind = np.argmin(abs(time-tstart)), np.argmin(abs(time-tstop))  # convert to indices

    # Slice data around burst
    new_stokes = full_stokes.copy()[:,tstart_ind:tstop_ind]  # slice data around burst
    time_burst = time.copy()[tstart_ind:tstop_ind]  # new time axis

    return new_stokes, time_burst


def invert_delay(delay, freq, U, V):
    """
    Invert a delay (e.g. from a burst search) in the Stokes U and V spectra.

    Parameters
    ----------
    delay : float or astropy Quantity
        The delay to invert (in ns or with time units).
    freq : array-like or astropy Quantity
        The frequency array (in MHz or with frequency units).
    U, V : 2D arrays
        Stokes U and V arrays (shape: [Ntimes, Nfreqs]).
    """
    # Assign units if not given
    if not isinstance(delay, u.quantity.Quantity):
        delay = delay * u.ns
    if not isinstance(freq, u.quantity.Quantity):
        freq = freq * u.MHz
    
    # Invert delay in U and V 
    arg = (2*np.pi * freq[None,:].to(1/u.s) * delay.to(u.s)).value  # x
    U_no_delay = U*np.cos(arg) + V*np.sin(arg)  # U_no_delay = U'cosx + V'sinx
    V_no_delay = -U*np.sin(arg) + V*np.cos(arg)  # V_no_delay = -U'sinx + V'cosx

    return U_no_delay, V_no_delay


def get_spectra(Q, U, freq,  t_range=[0,None], f_range=[0,None], normalize=True, replace_nans=True):
    """
    Calculate (normalized) Q and U spectra.
    Returns normalized Q and U spectra averaged over the specified time range.

    Parameters
    ----------
    Q : Stokes Q array (shape: [Nfreqs, Ntimes]).
    U : Stokes U array (shape: [Nfreqs, Ntimes]).
    freq : Frequency array (in MHz), matches Nfreqs.
    t_range : Time range to average over (in indices) [start, stop].
    f_range : Frequency range to get the spectrum for (in indices) [start, stop].
    normalize : If True, normalize the spectra by L.
    replace_nans : If True, replace NaNs with 0 in the normalized spectra.
    """
    
    # Time average
    f_start, f_stop = f_range
    freq = freq[f_start:f_stop]
    t_start, t_stop = t_range
    Q_spec = np.nanmean((Q)[f_start:f_stop,t_start:t_stop], axis=1)
    U_spec = np.nanmean((U)[f_start:f_stop,t_start:t_stop], axis=1)

    # Normalize
    if normalize:
        L_spec = np.sqrt(Q_spec**2 + U_spec**2)
        Q_spec_norm = Q_spec / L_spec
        U_spec_norm = U_spec / L_spec
    else:
        Q_spec_norm = Q_spec
        U_spec_norm = U_spec

    # Handle NaNs
    if replace_nans:
        Q_spec_norm = np.nan_to_num(Q_spec_norm, nan=0.0)
        U_spec_norm = np.nan_to_num(U_spec_norm, nan=0.0)

    return Q_spec_norm, U_spec_norm


def rm_synthesis(P_spec, phi_array, b, K):
    """
    Perform RM synthesis on the given P spectrum.
    
    Parameters
    ----------
    P_spec : Normalized complex P spectrum (shape: [Nfreqs,]).
    phi_array : Array of phi values to compute the FDF for (shape: [N_phi,]). In rad.
    b : Pre-computed exponential term for the FDF calculation (shape: [N_phi, Nfreqs]).
    K : Normalization constant for the FDF calculation.

    Returns
    -------
    RM_meas : Phi value at which the peak of the FDF occurs (in rad/m^2).
    FDF : F(phi) for the given P_spec (shape: [N_phi,]).
    """

    # Compute FDF
    # K = 1.0 / np.nansum(W)  # normalization constant
    # a = (-2.0 * 1j * phi_array).astype('complex64')
    # b = np.outer(a, lambda_sq)  # shape (Nphi, Nlambda2)
    FDF = K * np.sum(P_spec * b, 1)  # sum along lambda axis & normalize

    # Find phi where peak FDF occurs (= RM)
    FDF_peak_ind = np.argmax(np.abs(FDF))
    RM_meas = phi_array[FDF_peak_ind]

    return RM_meas, FDF