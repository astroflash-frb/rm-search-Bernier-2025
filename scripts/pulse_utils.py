"""
Functions for finding pulses in RM search results.
"""

import numpy as np
import astropy.units as u
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Default plot parameters
# ------------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 300,
    "figure.figsize": (8, 6),
    "font.size": 12,
    "font.family": "serif",
    "figure.titlesize": 16,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.grid": False,
    "savefig.bbox": "tight",
})


def count_visible_pulses(stokes_arr, prominence_cutoff=3, height_cutoff=2, max_dist=None):
    """
    Count visible pulses in the Stokes I timeseries by finding peaks above a certain prominence.

    Parameters
    ----------
    stokes_arr : np.ndarray
        Array of shape (Nstokes, Ntimes, Nfreqs) containing the Stokes parameters as a function of time and frequency.
    prominence_cutoff : float
        Minimum prominence of peaks to be counted as visible pulses.
    height_cutoff : float
        Minimum height of peaks to be counted as visible pulses.
    max_dist : int or None
        Minimum distance (in number of time bins) between peaks to be counted as separate pulses.

    Returns
    -------
    num_visible_pulses : int
        Number of visible pulses in Stokes I.
    peaks : list of ints
        Each element corresponds to the time index of a visible pulse. 
        **Multiple detections might belong to the same physical pulse (depending on max_dist).
    """
    
    # Get Stokes I snr timeseries
    stokes_I = stokes_arr[0]  # shape (Ntimes, Nfreqs)
    timeseries = np.nanmean(stokes_I, axis=1)  # Average over frequency
    median = np.nanmedian(timeseries)
    std = np.nanstd(timeseries)
    snr_timeseries = (timeseries - median) / std

    # Find peaks
    peaks, _ = find_peaks(snr_timeseries, 
                          prominence=prominence_cutoff, 
                          height=height_cutoff, 
                          distance=max_dist)  # indices of peaks in time_arr where peaks were detected
    num_visible_pulses = len(peaks)

    return num_visible_pulses, peaks


def downsample_time_mean(data, time, factor):
    """
    Downsample data along the time axis by some factor (number of time bins per group)
    """
    # Trim time axis so it's divisible by 'factor'
    Ntime = data.shape[0]
    Ntime_trimmed = (Ntime // factor) * factor
    trimmed_data = data[:Ntime_trimmed]
    trimmed_time = time[:Ntime_trimmed]

    # Group consecutive time samples into bins of size 'factor'
    # New shape: (Ngroups, factor, Nphi)
    grouped_data = trimmed_data.reshape(-1, factor, data.shape[1])
    grouped_time = trimmed_time.reshape(-1, factor)

    # Average within each group (collapse the 'factor' axis)
    downsampled_data = grouped_data.mean(axis=1)
    downsampled_time = grouped_time.mean(axis=1)
    
    return downsampled_data, downsampled_time


def separate_pulses(detections, time_tol):
    """
    Helper function to separate peak detections into distinct bursts.
    """

    bursts = []  # list of lift dicts: 1 element = 1 burst, dicts = all peaks found within that burst
    burst_start_time = None
    current_burst = []

    for det in detections: # go through all detected peaks
        if burst_start_time is None:
            burst_start_time = det["time"]
            current_burst = [det]
            continue

        dt = (det["time"] - burst_start_time) * u.s  # how far are we from the start of this pulse?
        if dt <= time_tol:  # still in same burst
            current_burst.append(det)  # append peaks to current burst info
        else:  # new burst! 
            bursts.append(current_burst)
            burst_start_time = det["time"]  # update start time
            current_burst = [det]  # start new burst

    # make sure we add last burst to list of bursts
    if len(current_burst):
        bursts.append(current_burst)
    
    return bursts


def plot_pulses(bursts, fdf, phi_arr, rm_true, save_name=None, save_loc=None):
    """
    Helper function to plot FDF slices with detected peaks for each burst, along with vertical lines at the true RM of the burst.
    """

    for i in range(len(bursts)):
        burst = bursts[i]
        plt.figure(figsize=(8,4), dpi=150)
        plt.title(f"{burst[0]['time']:.2f} - {burst[-1]['time']:.2f} s")
        plt.ylabel("S/N of FDF Amplitude")
        plt.xlabel(r'$\phi$ [rad/m$^2$]')

        # plot all slices with peaks detected
        for det in burst:
            t_ind = det["t_ind"]
            peaks = det["peaks"]
            fdf_line, = plt.plot(phi_arr, np.abs(fdf[t_ind]), c='k', lw=0.8, alpha=0.7, zorder=1)
            peak_marker = plt.scatter(phi_arr[peaks], np.abs(fdf[t_ind])[peaks], c='red', s=10, zorder=3)
            median_line = plt.axhline(y=det["median"], c='lightblue', ls='--', alpha=0.9, lw=1, zorder=3)

        # rm and median lines
        rm_line = plt.axvline(x=rm_true, c='orange', ls='-.', alpha=0.9, lw=1, zorder=2)
        neg_rm_line = plt.axvline(x=-rm_true, c='orange', ls=':', alpha=0.9, lw=1, zorder=2)

        # legend
        plt.legend(
            handles=[fdf_line, peak_marker, median_line, rm_line, neg_rm_line], 
            labels=['Data', 'Detected peaks', 'Median', 'True RM', '-RM'],
            loc="upper right")
        
        plt.xlim(np.min(phi_arr), np.max(phi_arr))

        # save figure
        if save_name is not None and save_loc is not None:
            plt.savefig(save_loc + save_name + f"_{burst[0]['time']:.2f}.png", dpi=150)
        plt.close()


def find_pulses_fdf(fdf, phi_arr, time_arr, downsamp_factor, time_tol, 
                    prominence_factor=10, height=2, rm_true=None, 
                    save_loc=None, save_name=None):
    """
    Wrapper function to find peaks in the FDF as a function of time, and separate them into distinct bursts based on a time tolerance.

    Returns
    -------
    bursts : list of lists of dicts
        Each element of the outer list corresponds to a detected burst, which is a list of dicts (one dict per time slice in which the burst was detected).
        Each dict contains info about the detected peaks in that time slice, ('t_ind', 'time', 'peaks', 'median').
        'peaks' are indices along the phi axis where peaks were detected in that time slice; 'median' is calculated along the phi axis for that time slice.
        Eg. [  [ {'t_ind': 0, 'time': 0.1, 'peaks': [10, 50], 'median': 5}, {'t_ind': 1, 'time': 0.2, 'peaks': [12], 'median': 4} ],  # burst with 2 detections
               [ {'t_ind': 10, 'time': 1.0, 'peaks': [30], 'median': 3} ]   # burst with 1 detection
            ]
    """

    # Mask out phi values between -10 and 10 rad/m^2 to avoid peak detection near RM=0
    phi_mask = np.abs(phi_arr) > 10 # exclude phi values between -10 and 10
    fdf = fdf.copy()
    fdf[:,~phi_mask] = np.nan

    # Downsample FDF along time axis
    downsampled_fdf, downsampled_time = downsample_time_mean(
        fdf,
        time_arr,
        downsamp_factor
    )

    detections = []  # list of all detections in FDF
    for t_ind, fdf_slice in enumerate(downsampled_fdf):

        # Calculate S/N of FDF slice along phi axis for this time slice
        fdf_slice = abs(fdf_slice) # (Nphi,)
        median = np.nanmedian(fdf_slice, axis=0)
        sigma = np.nanstd(fdf_slice, axis=0)    
        snr_slice = np.divide(  # to avoid /0
            fdf_slice - median,  # (fdf-median) / std
            sigma,
            out=np.zeros_like(fdf_slice),
            where=(sigma != 0) & ~np.isnan(fdf_slice)
        )  # shape (Nphi,)
        
        # Find peaks in S/N of phi distribution for this time slice
        peaks, _ = find_peaks(
            snr_slice, 
            prominence=prominence_factor, 
            height=height)

        if len(peaks) == 0:  # no peaks above S/N cutoff for this time slice
            continue

        detections.append({  # if detections -> append results
            't_ind': t_ind,
            'time': downsampled_time[t_ind],
            'peaks': peaks,
            'median': median,
        })
    
    # Separate detections into individual pulses based on time_tol + plot them
    bursts = separate_pulses(detections, time_tol)
    if save_loc is not None and save_name is not None:
        plot_pulses(bursts, downsampled_fdf, phi_arr, rm_true, 
                    save_loc=save_loc, save_name=save_name)

    return len(bursts), bursts



def count_found_notvisible(detected_bursts, visible_groups, time_arr_vis):
    """
    Count how many detected bursts don't appear visible in Stokes I.
    This uses detected bursts returned by find_pulses() and the visible pulse groups returned by count_visible_pulses().

    Parameters
    ----------
    detected_bursts : list of list dicts
        Each element of the outer list corresponds to a detected burst, which is a list of dicts (one dict per time slice in which the burst was detected).
        Each dict contains info about the detected peaks in that time slice, ('t_ind', 'time', 'peaks', 'median').
    visible_groups : list of lists of indices (each sublist corresponds to a visible pulse group in Stokes I).

    Returns
    -------
    num_notvisible : int
        Number of detected bursts that don't appear visible in Stokes I.
    notvisible_intervals : list of tuples
        List of (start_time, end_time) tuples for each detected burst that doesn't appear visible in Stokes I.
    """

    # Edge cases
    if len(detected_bursts) == 0:
        return 0
    if len(visible_groups) == 0:
        return len(detected_bursts)
    
    # Convert detected bursts to intervals
    detect_starts = np.array([
        burst[0]["time"] for burst in detected_bursts  # burst[0] is the first detection in a burst, 'time' gets the time value of that detection
    ])
    detect_ends = np.array([
        burst[-1]["time"] for burst in detected_bursts  # burst[-1] is the last detection in a burst, 'time' gets the time value of that detection
    ])

    # Convert visible groups to intervals
    visible_times = time_arr_vis[np.array(visible_groups)]

    # single visible interval (with tolerance padding of 0.05s=50ms on either side)
    visible_starts = visible_times - 0.05
    visible_ends = visible_times + 0.05

    # Check overlaps using broadcasting
    overlap = (
        detect_starts[:, None] <= visible_ends[None, :]
    ) & (
        visible_starts[None, :] <= detect_ends[:, None]
    )

    # FDF bursts with no detection in Stokes I 
    found_notvisible = ~np.any(overlap, axis=1)
    notvisible_intervals = list(zip(
        detect_starts[found_notvisible],
        detect_ends[found_notvisible]
    ))

    return np.sum(found_notvisible), notvisible_intervals

