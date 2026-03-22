"""
Perform an RM search on the given full Stokes data.
"""

import argparse
from time import perf_counter, process_time
import os
import psutil
import resource
from pathlib import Path
import numpy as np
from astropy.constants import c
import astropy.units as u
from scipy.signal import correlate, correlation_lags
import chime_frb_constants as constants


# Get Arguments
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

# Required
parser.add_argument("data_file", type=Path, 
                    help='Absolute path to .npz data file containing full_stokes, time, and frequency.')
parser.add_argument("save_file", type=Path, 
                    help='Absolute path to save .npz file with RM search results.')

# Optional
parser.add_argument(
    "-s", 
    "--sim_file", 
    type=Path, 
    default=None, 
    help='File of simulated stokes parameters to inject bursts into data.'
)
parser.add_argument(
    "-p", 
    "--phi_max", 
    type=int, 
    default=None, 
    help='Maximum phi value for RM synthesis.'
)
parser.add_argument(
    "-d", 
    "--dphi_scaling", 
    type=float, 
    default=0.1, 
    help='Scaling factor for phi spacing in dphi = scaling * FWHM (default: 0.1).'
)
parser.add_argument(
    "--rfi_mean_threshold", 
    type=float, 
    default=2.0, 
    help='Threshold for mean-based RFI flagging (default: 2.0).'
)
parser.add_argument(
    "--rfi_std_threshold", 
    type=float, 
    default=5.0, 
    help='Threshold for std-based RFI flagging (default: 5.0).'
)
parser.add_argument(
    "--search_time_step", 
    type=int, 
    default=10, 
    help='Number of time channels to average over during RM search (default: 10).'
)

args = parser.parse_args()

# Assign arguments to variables
DATA_FILE = args.data_file
SAVE_FILE = args.save_file
SIM_FILE = args.sim_file
PHI_MAX = args.phi_max
DPHI_SCALING = args.dphi_scaling
RFI_MEAN_THRESHOLD = args.rfi_mean_threshold
RFI_STD_THRESHOLD = args.rfi_std_threshold
SEARCH_TIME_STEP = args.search_time_step

# Other Constants & Globals
RFI_RANGES = [(529,535), (483, 483), (452,452)]  # in MHz
SAMPLE_RATE = constants.FPGA_COUNTS_PER_SECOND * u.Hz
TIMINGS = {}
_process = psutil.Process(os.getpid())



# Functions
# ------------------------------------------------------------------------------
def flag_rfi_manual(ranges, freqs):
    """
    ranges : list of tuples
        list of (start,end) frequencies to mask
    """
    channel_mask_manual = np.full(len(freqs), False)
    
    for (start,end) in ranges:
        # Check that freqs to mask are at least particlaly contained in the frequency array
        if start > freq[-1] or end < freqs[0]:  # completely out of range
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
    
    return (data_array - mean_per_channel) / std_per_channel


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


def rm_synthesis(P_spec, phi_array, b):
    """
    Perform RM synthesis on the given P spectrum.
    
    Parameters
    ----------
    P_spec : Normalized complex P spectrum (shape: [Nfreqs,]).
    phi_array : Array of phi values to compute the FDF for (shape: [N_phi,]). In rad.
    b : Pre-computed exponential term for the FDF calculation (shape: [N_phi, Nfreqs]).

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


def measure_start(name):
    """Start timing a block of code."""

    TIMINGS[name] = {
        "wall_start": perf_counter(),  # Start wall clock time
        "cpu_start": process_time(),  # Start CPU time (only counts time when CPU is actively working on the process)
        "mem_start_gb": _process.memory_info().rss / 1e9  # “How much RAM is my job currently using on the node?”
        # Times are in seconds, memory in GB
    }

def measure_stop(name):
    """Stop timing a code block and compute elapsed times."""

    # Make sure we started the timer for this block
    if name not in TIMINGS or "wall_start" not in TIMINGS[name]:
        raise ValueError(f"measure_start() not called for '{name}'")
    
    # Get end times and memory usage
    wall_end = perf_counter()  # seconds
    cpu_end = process_time()  # seconds

    # Get peak memory usage (only for total timing -- doesn't matter otherwise)
    mem_end_gb = _process.memory_info().rss / 1e9  # GB
    if name=='total':
        peak_mem_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024  # GB
    else:
        peak_mem_gb = None

    # Compute elapsed times and efficiency
    wall_elapsed = wall_end - TIMINGS[name]["wall_start"]
    cpu_elapsed = cpu_end - TIMINGS[name]["cpu_start"]

    # Add to TIMINGS dictionary
    TIMINGS[name].update({
        "wall_elapsed": wall_elapsed,
        "cpu_elapsed": cpu_elapsed,
        "cpu_efficiency": cpu_elapsed / wall_elapsed if wall_elapsed > 0 else 0,
        "mem_end_gb": mem_end_gb,
        "mem_peak_gb": peak_mem_gb
    })



if __name__ == "__main__":
    # Start total timing
    measure_start("total")

    # Open & Process Data
    # ------------------------------------------------------------------------------
    measure_start("load_mask_normalize")

    # Load data
    full_stokes_npz = np.load(DATA_FILE)

    full_stokes = full_stokes_npz['full_stokes']   # shape (Nstokes, Ntimes, Nfreqs)
    time = full_stokes_npz['time']
    freq = full_stokes_npz['freq']

    # Get RFI mask using intensity data
    rfi_mask = get_rfi_mask(full_stokes[0], 
                            mean_treshhold=RFI_MEAN_THRESHOLD, 
                            std_threshold=RFI_STD_THRESHOLD,
                            ranges=RFI_RANGES, freqs=freq)

    # Normalize & mask RFI channels
    stokes_norm_masked = normalize_data(full_stokes.copy())
    stokes_norm_masked[:,:,rfi_mask] = np.nan

    # Inject bursts from sim file, if provided
    sim_params = None
    if SIM_FILE is not None:
        sim_data = np.load(SIM_FILE, allow_pickle=True)
        stokes_sim = sim_data['full_stokes']  # shape (Nstokes, Ntimes, Nfreqs)
        stokes_norm_masked += stokes_sim
        sim_params = sim_data['params']

    measure_stop("load_mask_normalize")


    # RMSF
    # ------------------------------------------------------------------------------
    measure_start("rmsf_computation")

    # Define weight function (with and without masked channels)
    W_full = np.ones_like(freq)
    W = W_full.copy()
    W[rfi_mask] = 0  # Set bad channels to zero

    # lambda^2 array
    lambda2_array = (c.value/(freq*1e6))**2  # Squared wavelength array (shape: [Nfreqs,]). In m^2.
    l2_min = np.min(lambda2_array)
    l2_max = np.max(lambda2_array)
    dl2 = np.median(np.abs(np.diff(lambda2_array)))
    Dl2 = l2_max - l2_min  # total range of lambda^2

    # phi array
    fwhm = 2 * np.sqrt(3) / Dl2  # FWHM of the RMSF
    if PHI_MAX is None:
        PHI_MAX = np.sqrt(3)/dl2
        # phi_max = np.sqrt(3)/dl2  # max RM to search
        # phi_max = 10 * fwhm  # ~10*FWHM
    dphi = DPHI_SCALING * fwhm  # spacing between phi values, ~0.1*FWHM
    phi_array = np.arange(-PHI_MAX, PHI_MAX + dphi, dphi)  # shape (N_phi,)
    N_phi = len(phi_array)

    # Compute FSF (RMSF) w/ masked channels
    K = 1.0 / np.nansum(W)  # normalization constant
    a = (-2.0 * 1j * phi_array).astype('complex64')  # exponent for RMSF
    b = np.outer(a, lambda2_array)  # shape (Nphi, Nlambda2)
    RMSF = K * np.sum(W * np.exp(b), 1)  # sum along lambda axis & normalize

    # RMSF w/ all channels
    K_full = 1.0 / np.sum(W_full)  # normalization constant
    RMSF_full = K_full * np.sum(W_full * np.exp(b), 1)  # sum along lambda axis & normalize

    measure_stop("rmsf_computation")


    # RM Search
    # ------------------------------------------------------------------------------
    measure_start("rm_search")

    # Split time array into slices of length SEARCH_TIME_STEP
    time_slice_arr = time[::SEARCH_TIME_STEP]  # New time array (using start times of each time slice)

    # Set up arrays for results
    RM_meas_arr = np.zeros(len(time_slice_arr))  # Store peak of FDF for each time slice
    FDF_arr = np.zeros((len(time_slice_arr), N_phi), dtype=complex)  # FDF for each time slice

    # Pre-compute the exponential for the FDF calculation
    K = 1.0 / np.nansum(W)  # normalization constant
    a = (-2.0 * 1j * phi_array).astype('complex64')
    b = np.exp(np.outer(a, lambda2_array))  # shape (Nphi, Nlambda2)

    # Step through time & compute FDF
    for i,start_time in enumerate(time_slice_arr):
        # Time slice indices
        t_start_ind = np.argwhere(time==start_time)[0][0]
        t_stop_ind = t_start_ind + SEARCH_TIME_STEP

        # Get the spectra (avg over time slice)
        Q_spec_norm, U_spec_norm = get_spectra(stokes_norm_masked[1].T, 
                                               stokes_norm_masked[2].T, 
                                               freq=freq, 
                                               t_range=[t_start_ind, t_stop_ind]
                                               )
        P_spec_norm = Q_spec_norm + 1j*U_spec_norm

        # Compute FDF and RM for this time slice
        RM_meas_arr[i], FDF_arr[i] = rm_synthesis(P_spec_norm,
                                                  phi_array=phi_array,
                                                  b=b
                                                )
    
    measure_stop("rm_search")
        
    
    # Cross-Correlation
    # ------------------------------------------------------------------------------
    measure_start("cross_correlation")

    # Define array to hold cross-correlation results (shape: [Ntimes, Nphi])
    cross_corr_len = 2 * len(phi_array) - 1
    cross_corr_arr = np.zeros((len(time_slice_arr), cross_corr_len))

    # Cross-correlate measured FDF with FSF for each time slice
    for i,t in enumerate(time_slice_arr):
        # Get absolute value of FDF for the time slice
        fdf_slice = np.abs(FDF_arr[i])
        
        # Cross-correlate with FSF
        cross_corr_arr[i] = correlate(fdf_slice, np.abs(RMSF), mode='full')

    # Get channel lags and corresponding phi values
    channel_lags = correlation_lags(len(phi_array), len(phi_array), mode='full')
    phi_lags = channel_lags * dphi

    measure_stop("cross_correlation")
    

    # Save Results & Metadata
    # ------------------------------------------------------------------------------
    # Stop total timing
    measure_stop("total")

    # Save data arrays
    arrays_file = SAVE_FILE.with_name(SAVE_FILE.stem + "_arrays.npz")
    np.savez_compressed(
        arrays_file,   # filename
        stokes_norm_masked=stokes_norm_masked,  # shape (Nstokes, Ntimes, Nfreqs)
        time=time,  # shape (Ntimes,)
        freq=freq,  # shape (Nfreqs,)
        rfi_mask=rfi_mask,  # shape (Nfreqs,)
        RMSF=RMSF,  # shape (Nphi,)
        RMSF_full=RMSF_full,  # shape (Nphi,)
        lambda2_array=lambda2_array,  # shape (Nlambda,) = (Nfreq,)
        phi_array=phi_array,  # shape (Nphi,)
        FDF_arr=FDF_arr,  # shape (Ntimes_fdf, Nphi)
        RM_meas_arr=RM_meas_arr, # phi where peak of FDF occurs, for each time slice. shape (Ntimes_fdf,)
        time_slice_arr=time_slice_arr,  # shape (Ntimes_fdf,)
        cross_corr_arr=cross_corr_arr,  # shape (Ntimes_fdf, Nphi)
        phi_lags=phi_lags  # shape (Nlags,)
    )
    print(f"Saved RM search arrays to {arrays_file}")

    # Compute some additional metrics for metadata
    nsamples_per_chunk = int(DATA_FILE.name.split("_")[2][7:])   # num channels averaging from voltages->stokes
    dt_stokes = (nsamples_per_chunk/SAMPLE_RATE).to(u.ms)  # time resolution of the Stokes data, in ms
    df_stokes = (400/1024) * u.MHz  # frequency resolution, 1024 channels across 400 MHz bandwidth

    # Build metadata
    metadata = {
        # Data file info
        'source': DATA_FILE.parent.name,
        'night': DATA_FILE.name.split("_")[0],
        'freq_bands': [int(f) for f in DATA_FILE.name.split("_")[1][5:]],
        'fmin': freq[0],
        'fmax': freq[-1],
        'df_stokes': df_stokes,  # frequency resolution of the Stokes data, in MHz
        'nsamples_per_chunk': nsamples_per_chunk,  # number of channels averaged together to get the Stokes data from the voltages
        'dt_stokes': dt_stokes,  # time resolution of the Stokes data, in ms
        'delta_t_total': time[-1] - time[0],  # total time duration of the data, in seconds
        'nfiles': int(DATA_FILE.name.split("_")[3][:3]),  # number of files read in
        'start_file_ind': int(DATA_FILE.name.split("_")[4][5:-4]),  # index of first fild read (relative to raw baseband files)
        # RM search parameters
        'phi_max': PHI_MAX,
        'dphi_scaling': DPHI_SCALING,  # dphi = scaling * FWHM
        'rfi_mean_threshold': RFI_MEAN_THRESHOLD,
        'rfi_std_threshold': RFI_STD_THRESHOLD,
        'search_time_step': SEARCH_TIME_STEP,  # number of time *channels* averaged over during RM search
        'dt_rm': SEARCH_TIME_STEP * dt_stokes,  # effective time resolution of RM search results in ms
        # SLURM info
        'slurm_info': {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "num_nodes": int(os.environ.get("SLURM_JOB_NUM_NODES", 1)),
            "ntasks": int(os.environ.get("SLURM_NTASKS", 1)),
            "cpus_per_task": int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
        },
        # Timings
        'timings': TIMINGS,
        # Injected bursts info
        'sim_params': sim_params  # parameters of injected bursts from sim file, if provided
    }

    # Print metrics
    print("\nRM Search Metrics:")
    for key, value in metadata.items():
        print(f"{key}: {value}")

    # Save metadata
    metadata_file = SAVE_FILE.with_name(SAVE_FILE.stem + "_metadata.npz")
    np.savez(metadata_file, metadata=metadata)  # will need to unpack the metadata dict when opening
    print(f"Saved metadata to {metadata_file}")
