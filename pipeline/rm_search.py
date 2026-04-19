"""
Perform an RM search on the given full Stokes data.
"""

from rm_search_utils import *  # contains all functions related to masking RFI, normalizing, and computing spectra and FDF
from sim_burst_utils import *  # contains all functions related to generating and injecting simulated bursts
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
parser.add_argument(
    "--sim_flag",
    type=int,
    default=0,
    help='Set to 1 to inject simulated bursts, 0 to run without (default: 0).'
)
parser.add_argument(
    "--n_bursts",
    type=int,
    default=0,
    help='Number of simulated bursts to inject if sim_flag=1 (default: 2).'
)
parser.add_argument(
    "--sim_arrival_times",
    type=float,
    nargs='+',
    default=None,
    help='List of times (in seconds) at which to inject simulated bursts, if sim_flag=1. Must be contained within the time range of the input data.'
)
parser.add_argument(
    "--sim_burst_widths",
    type=float,
    nargs='+',
    default=None,
    help='List of widths (in seconds) for the simulated bursts to inject, if sim_flag=1.'
)
parser.add_argument(
    "--sim_snr",
    type=float,
    nargs='+',
    default=None,
    help='List of SNRs for the simulated bursts to inject (relative to noise in real timeseries data), if sim_flag=1.'
)
parser.add_argument(
    "--sim_rm",
    type=float,
    nargs='+',
    default=None,
    help='List of RMs (in rad/m^2) for the simulated bursts to inject, if sim_flag=1.'
)

args = parser.parse_args()

# Files
DATA_FILE = args.data_file
SAVE_FILE = args.save_file

# Search parameters
PHI_MAX = args.phi_max
DPHI_SCALING = args.dphi_scaling
RFI_MEAN_THRESHOLD = args.rfi_mean_threshold
RFI_STD_THRESHOLD = args.rfi_std_threshold
SEARCH_TIME_STEP = args.search_time_step

# Simulation parameters 
SIM_FLAG = args.sim_flag
N_BURSTS = args.n_bursts
SIM_ARRIVAL_TIMES = args.sim_arrival_times
SIM_BURST_WIDTHS = args.sim_burst_widths
SIM_SNR = args.sim_snr
SIM_RM = args.sim_rm
SIM_PARAMS = None  # to be filled with sim params if SIM_FLAG=1

# Other Constants & Globals
RFI_RANGES = [(529,535), (482,483),(450,450),(452,452),(457,458),(462,467),(470,470),(477,477)]  # in MHz
SAMPLE_RATE = constants.FPGA_COUNTS_PER_SECOND * u.Hz
TIMINGS = {}
_process = psutil.Process(os.getpid())



# Timing Functions
# ------------------------------------------------------------------------------
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

    # Open & Process Data (load, add sim bursts, mask RFI, normalize)
    # ------------------------------------------------------------------------------
    measure_start("load_mask_normalize")

    # Load data
    full_stokes_npz = np.load(DATA_FILE)
    full_stokes = full_stokes_npz['full_stokes']   # shape (Nstokes, Ntimes, Nfreqs)
    time = full_stokes_npz['time']
    freq = full_stokes_npz['freq']

    # Inject bursts
    if SIM_FLAG:
        # Burst parameters
        burst_params = gen_sim_burst_params(N_BURSTS, SIM_ARRIVAL_TIMES, SIM_BURST_WIDTHS, freq)
        
        # Pol fractions and angle
        pol_frac_linear = [0.7, 0.4]  # can also have diff fractions for U and Q
        pol_frac_circular = 0.
        psi_0_rad = np.radians(30)  # Intrinsic polarization angle, [rad]

        sigma_time = compute_timeseries_sigma(full_stokes)  # noise level in time series

        # Build dict to save sim metadata
        SIM_PARAMS = {
            'num_bursts': N_BURSTS,
            'arrival_times': SIM_ARRIVAL_TIMES,
            'burst_widths': SIM_BURST_WIDTHS,
            'snr': SIM_SNR,
            'rm': SIM_RM,
            'pol_frac_linear': pol_frac_linear,
            'pol_frac_circular': pol_frac_circular,
            'psi_0_rad': psi_0_rad, 
            'sigma_time': sigma_time
        }

        # Generate Stokes
        # component shape is (Nbursts, Nfreqs, Ntimes); sim shape is (Nfreqs, Ntimes)
        I_components = gen_stokes_I(N_BURSTS, burst_params, freq, time, sigma_time, SIM_SNR)
        I_sim, Q_components, U_components, V_sim = gen_stokes_QUV(I_components, 
                                                                  N_BURSTS, 
                                                                  pol_frac_linear, 
                                                                  pol_frac_circular, 
                                                                  psi_0_rad)
        
        # Add rotation measure & get full stokes array
        Q_rot, U_rot = get_rot_stokes(I_components, Q_components, U_components, N_BURSTS, SIM_RM, freq)
        full_stokes_sim = np.array([I_sim.T, Q_rot.T, U_rot.T, V_sim.T])  # shape (Nstokes, Ntimes, Nfreqs)

        # Inject bursts in real data
        full_stokes += full_stokes_sim

    # Get RFI mask using intensity data
    rfi_mask = get_rfi_mask(full_stokes[0], 
                            mean_treshhold=RFI_MEAN_THRESHOLD, 
                            std_threshold=RFI_STD_THRESHOLD,
                            ranges=RFI_RANGES, freqs=freq)

    # Normalize & mask RFI channels
    stokes_norm_masked = normalize_data(full_stokes.copy())
    stokes_norm_masked[:,:,rfi_mask] = np.nan

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
        PHI_MAX = np.sqrt(3)/dl2  # max RM to search
        # phi_max = 10 * fwhm  # ~10*FWHM
    dphi = DPHI_SCALING * fwhm  # spacing between phi values, ~0.1*FWHM
    phi_array = np.arange(-PHI_MAX, PHI_MAX + dphi, dphi)  # shape (N_phi,)
    N_phi = len(phi_array)

    # Compute FSF (RMSF) w/ masked channels
    K = 1.0 / np.nansum(W)  # normalization constant
    a = (-2.0 * 1j * phi_array).astype('complex64')  # -2i phi
    b = np.exp(np.outer(a, lambda2_array))  # e^{-2i phi lambda^2}, shape (Nphi, Nlambda2)
    RMSF = K * np.sum(W * b, 1)  # sum along lambda axis & normalize

    # RMSF w/ all channels
    K_full = 1.0 / np.sum(W_full)  # normalization constant
    RMSF_full = K_full * np.sum(W_full * b, 1)  # sum along lambda axis & normalize

    measure_stop("rmsf_computation")


    # RM Search
    # ------------------------------------------------------------------------------
    measure_start("rm_search")

    # Split time array into slices of length SEARCH_TIME_STEP
    time_slice_arr = time[::SEARCH_TIME_STEP]  # New time array (using start times of each time slice)

    # Set up arrays for results
    RM_meas_arr = np.zeros(len(time_slice_arr))  # Store peak of FDF for each time slice
    FDF_arr = np.zeros((len(time_slice_arr), N_phi), dtype=complex)  # FDF for each time slice

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
                                                  b=b, K=K
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

    # Get correct NFILES
    nfiles = DATA_FILE.name.split("_")[3][:3]  # number of files read in (may have trailing f if <100)
    if nfiles[-1] == 'f' : 
        nfiles = nfiles[:-1]  # remove trailing 'f' if present
    nfiles = int(nfiles)

    # Build metadata
    metadata = {
        # Data file info
        'source': DATA_FILE.parent.name,
        'night': DATA_FILE.name.split("_")[0],
        'freq_bands': [int(f) for f in DATA_FILE.name.split("_")[1][5:]],
        'fmin': freq[0],
        'fmax': freq[-1],
        'df_stokes_MHz': df_stokes,  # frequency resolution of the Stokes data, in MHz
        'nsamples_per_chunk': nsamples_per_chunk,  # number of channels averaged together to get the Stokes data from the voltages
        'dt_stokes_ms': dt_stokes,  # time resolution of the Stokes data, in ms
        'delta_t_total_s': time[-1] - time[0],  # total time duration of the data, in seconds
        'nfiles': nfiles,  # number of files read in
        'start_file_ind': int(DATA_FILE.name.split("_")[4][5:-4]),  # index of first fild read (relative to raw baseband files)
        # RM search parameters
        'phi_max': PHI_MAX,
        'dphi_scaling': DPHI_SCALING,  # dphi = scaling * FWHM
        'rfi_mean_threshold': RFI_MEAN_THRESHOLD,
        'rfi_std_threshold': RFI_STD_THRESHOLD,
        'search_time_step': SEARCH_TIME_STEP,  # number of time *channels* averaged over during RM search
        'dt_rm_ms': SEARCH_TIME_STEP * dt_stokes,  # effective time resolution of RM search results in ms
        # SLURM info
        'slurm_info': {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "num_nodes": int(os.environ.get("SLURM_JOB_NUM_NODES", 1)),
            "ntasks": int(os.environ.get("SLURM_NTASKS", 1)),
            "cpus_per_task": int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
        },
        # Injected bursts info
        'sim_params': SIM_PARAMS  # parameters of injected bursts
    }

    # Print metadata
    print("\nRM Search Metadata:")
    for key, value in metadata.items():
        print(f"{key}: {value}")

    # Save metadata
    metadata_file = SAVE_FILE.with_name(SAVE_FILE.stem + "_metadata.npz")
    np.savez(metadata_file, metadata=metadata)  # will need to unpack the metadata dict when opening
    print(f"Saved metadata to {metadata_file}")

    # Save timing results
    timing_file = SAVE_FILE.with_name(SAVE_FILE.stem + "_timing.npz")
    np.savez(timing_file, timings=TIMINGS)  # will need to unpack the timings dict when opening
    print(f"Saved timing results to {timing_file}")
