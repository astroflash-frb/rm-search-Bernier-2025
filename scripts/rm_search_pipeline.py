"""
Perform an RM search on the given data.
"""

from search_utils import *  # contains all functions related to masking RFI, normalizing, and computing spectra and FDF
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

# Required (paths)
parser.add_argument("source_dir", type=Path, 
                    help='Absolute path to directory containing data of the source to analyze.')
parser.add_argument("obs_night", type=str, 
                    help='Observation night to analyze.')
parser.add_argument("save_file", type=Path, 
                    help='Absolute path to save .npz file with RM search results.')

# Optional
# Source properties
parser.add_argument(
    "--true_dm",
    type=float,
    default=0.0,
    help='DM value (in pc/cm**3) to use when dedispersing the data read in (default: 0.0).'
)
parser.add_argument(
    "--true_rms",
    type=float,
    nargs='+',
    default=None,
    help='True RM values for data read in.'
)
parser.add_argument(
    "--delay",
    type=float,
    default=0,
    help='Time delay (in ns) to remove from the data read in. Default is 0 (no delay).'
)

# Data Parameters
parser.add_argument(
    "--start_file_ind", 
    type=int, 
    default=0, 
    help='Index of first file to read in (relative to raw baseband files, which are indexed in order of time). Default is 0.'
)
parser.add_argument(
    "--nfiles", 
    type=int, 
    default=65, 
    help='Number of files to read in (each file contains 50000 time samples). Default is 65.'
)
parser.add_argument(
    "--ref_freq", 
    type=float, 
    default=600.0, 
    help='Reference frequency (in MHz) to use when dedispersing the data (default: 600.0 MHz).'
)
parser.add_argument(
    "--npixels_to_avg", 
    type=int, 
    default=391, 
    help='Number of time bins to average over when reading in data (default: 391 = 1 ms).'
)
parser.add_argument(
    "--slice_burst_flag",
    type=int,
    default=0,
    help='Set to 1 to slice the data around the highest SNR burst, 0 to use the full data saved by baseband formatter (default: 0).'
)

# RM Search Parameters
parser.add_argument(
    "--phi_max", 
    type=int, 
    default=None, 
    help='Maximum phi value for RM synthesis.'
)
parser.add_argument(
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
    default=1, 
    help='Number of time channels to average over during RM search (default: 1).'
)

# Simulation parameters
parser.add_argument(
    "--sim_flag",
    type=int,
    default=0,
    help='Set to 1 to inject simulated bursts, 0 to run without (default: 0).'
)
parser.add_argument(
    "--sim_num_bursts",
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

# File info
SOURCE_DIR = args.source_dir
OBS_NIGHT = args.obs_night
SAVE_FILE = args.save_file
DM = pb.DispersionMeasure(args.true_dm * u.pc/u.cm**3)
RM = args.true_rms * u.rad/u.m**2
DELAY = args.delay * u.ns

# Data info
START_FILE_IND = args.start_file_ind
NFILES = args.nfiles
START_FRAME = START_FILE_IND * 50000
NUMBER_OF_FRAMES = NFILES * 50000  # Nfiles * Nframes_per_file
REF_FREQ = args.ref_freq * u.MHz  # reference frequency for dedispersion, in MHz
NPIXELS_TO_AVG = args.npixels_to_avg
SLICE_BURST_FLAG = args.slice_burst_flag

# Search parameters
PHI_MAX = args.phi_max
DPHI_SCALING = args.dphi_scaling
RFI_MEAN_THRESHOLD = args.rfi_mean_threshold
RFI_STD_THRESHOLD = args.rfi_std_threshold
SEARCH_TIME_STEP = args.search_time_step

# Simulation parameters 
SIM_FLAG = args.sim_flag
SIM_NUM_BURSTS = args.sim_num_bursts
SIM_ARRIVAL_TIMES = args.sim_arrival_times
SIM_BURST_WIDTHS = args.sim_burst_widths
SIM_SNR = args.sim_snr
SIM_RM = args.sim_rm
SIM_PARAMS = None  # to be filled with sim params if SIM_FLAG=1

# Other Constants & Globals
RFI_RANGES = [(529,535), (482,483),(450,450),(452,452),(457,458),(462,467),(470,470),(477,477)]  # in MHz
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

    # Open Data
    # ------------------------------------------------------------------------------
    measure_start("read_and_dedisperse")
    full_stokes, time, freq = read_stokes(SOURCE_DIR, OBS_NIGHT+"_CHIME_vdif", 
                                          START_FRAME, NUMBER_OF_FRAMES, 
                                          DM, REF_FREQ, 
                                          NPIXELS_TO_AVG)
    print(f"Data shape (Nstokes, Ntimes, Nfreqs): {full_stokes.shape}")
    print(f"Time array shape: {time.shape}, Time range: {time[0]:.3f} - {time[-1]:.3f} s")
    print(f"Frequency array shape: {freq.shape}, Frequency range: {freq[0]:.2f} - {freq[-1]:.2f} MHz")
    measure_stop("read_and_dedisperse")
 

    # Process Data (add sim bursts, mask RFI, normalize)
    # ------------------------------------------------------------------------------
    measure_start("mask_and_normalize")

    # Inject bursts
    if SIM_FLAG:
        # Burst parameters
        burst_params = gen_sim_burst_params(SIM_NUM_BURSTS, SIM_ARRIVAL_TIMES, SIM_BURST_WIDTHS, freq)
        
        # Pol fractions and angle
        pol_frac_linear = [0.7, 0.4]  # can also have diff fractions for U and Q
        pol_frac_circular = 0.
        psi_0_rad = np.radians(30)  # Intrinsic polarization angle, [rad]

        sigma_timeseries = compute_timeseries_sigma(full_stokes)  # noise level in time series

        # Build dict to save sim metadata
        SIM_PARAMS = {
            'num_bursts': SIM_NUM_BURSTS,
            'arrival_times': SIM_ARRIVAL_TIMES,
            'burst_widths': SIM_BURST_WIDTHS,
            'snr': SIM_SNR,
            'rm': SIM_RM,
            'pol_frac_linear': pol_frac_linear,
            'pol_frac_circular': pol_frac_circular,
            'psi_0_rad': psi_0_rad, 
            'sigma_timeseries': sigma_timeseries
        }

        # Generate Stokes
        # component shape is (Nbursts, Nfreqs, Ntimes); sim shape is (Nfreqs, Ntimes)
        I_components = gen_stokes_I(SIM_NUM_BURSTS, burst_params, freq, time, sigma_timeseries, SIM_SNR)
        I_sim, Q_components, U_components, V_sim = gen_stokes_QUV(I_components, 
                                                                  SIM_NUM_BURSTS, 
                                                                  pol_frac_linear, 
                                                                  pol_frac_circular, 
                                                                  psi_0_rad)
        
        # Add rotation measure & get full stokes array
        Q_rot, U_rot = get_rot_stokes(I_components, Q_components, U_components, SIM_NUM_BURSTS, SIM_RM, freq)
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

    # If slice_burst_flag is set, slice the data around the highest SNR burst
    if SLICE_BURST_FLAG:
        stokes_norm_masked, time = slice_around_burst(stokes_norm_masked, time)

    # Invert delay if given
    if DELAY != 0*u.ns:
        stokes_norm_masked[1], stokes_norm_masked[2] = invert_delay(DELAY, freq, stokes_norm_masked[1], stokes_norm_masked[2])

    measure_stop("mask_and_normalize")


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
    pos_phi_arr = np.arange(0, PHI_MAX+dphi, dphi)
    neg_phi_arr = np.flip(-pos_phi_arr[1:])
    phi_array = np.concatenate([neg_phi_arr, pos_phi_arr])  # shape (N_phi,)
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

    # Define output paths
    summary_file = SAVE_FILE.with_name(SAVE_FILE.stem + "_summary.txt")
    arrays_file = SAVE_FILE.with_name(SAVE_FILE.stem + "_arrays.npz")
    metadata_file = SAVE_FILE.with_name(SAVE_FILE.stem + "_metadata.npz")
    timing_file = SAVE_FILE.with_name(SAVE_FILE.stem + "_timing.npz")

    # Compute some additional metrics for metadata
    dt_stokes = (NPIXELS_TO_AVG / (constants.FPGA_COUNTS_PER_SECOND * u.Hz)).to(u.ms)  # time resolution of the Stokes data, in ms
    df_stokes = (400 / 1024) * u.MHz  # frequency resolution, 1024 channels across 400 MHz bandwidth

    # Build metadata
    metadata = {
        # Input info
        'source_dir': str(SOURCE_DIR),
        'obs_night': OBS_NIGHT,
        'save_file': str(SAVE_FILE),

        # Data properties
        'true_dm_pc_cm3': DM,
        'true_rms_rad_m2': RM,
        'delay_ns': DELAY,

        # Data parameters
        'start_file_ind': START_FILE_IND,       # index of first fild read
        'nfiles': NFILES,                       # number of files read in
        'start_frame': START_FRAME,             # start file converted to a frame number
        'number_of_frames': NUMBER_OF_FRAMES,   # number of frames read in (corresponding to nfiles)
        'ref_freq_MHz': REF_FREQ,               # reference frequency for dedispersion, in MHz
        'npixels_to_avg': NPIXELS_TO_AVG,       # number of time bins averaged together to get the Stokes data from the voltages
        'slice_burst_flag': SLICE_BURST_FLAG,

        # Data dimensions
        'n_time': len(time),                    # number time samples in Stokes
        'n_freq': len(freq),
        'n_phi': len(phi_array),
        'n_time_rm': len(time_slice_arr),       # number time samples after averaging for RM search (length of time_slice_arr)

        # Frequency / time info
        'fmin_MHz': float(np.nanmin(freq)),
        'fmax_MHz': float(np.nanmax(freq)),
        'df_stokes_MHz': float(df_stokes.value),        # frequency resolution of the Stokes data, in MHz
        'dt_stokes_ms': float(dt_stokes.value),         # time resolution of the Stokes data, in ms
        'delta_t_total_s': float(time[-1] - time[0]),   # total time duration of the data, in seconds

        # RM search parameters
        'phi_max_rad_m2': np.max(phi_array),    # maximum phi value used in RM synthesis
        'dphi_scaling': DPHI_SCALING,           # phi scaling when getting phi spacing
        'dphi_rad_m2': dphi,                    # dphi = scaling * FWHM
        'fwhm_rmsf_rad_m2': fwhm,               # fwhm of the RMSF in rad/m^2
        'search_time_step': SEARCH_TIME_STEP,   # number of time *channels* averaged over during RM search
        'dt_rm_ms': float((SEARCH_TIME_STEP * dt_stokes).value),  # effective time resolution of RM search results in ms

        # RFI masking
        'rfi_mean_threshold': RFI_MEAN_THRESHOLD,
        'rfi_std_threshold': RFI_STD_THRESHOLD,
        'num_masked_channels': int(np.sum(rfi_mask)),
        'fraction_masked_channels': float(np.sum(rfi_mask) / len(rfi_mask)),

        # Simulation parameters
        'sim_flag': SIM_FLAG,
        'sim_params': SIM_PARAMS,  # parameters of injected bursts

        # SLURM info
        'slurm_info': {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "num_nodes": int(os.environ.get("SLURM_JOB_NUM_NODES", 1)),
            "ntasks": int(os.environ.get("SLURM_NTASKS", 1)),
            "cpus_per_task": int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
        }
    }

    # -- Save data arrays --
    np.savez_compressed(
        arrays_file,                                        # filename
        full_stokes=stokes_norm_masked.astype(np.float32),  # shape (Nstokes, Ntimes, Nfreqs)
        time=time.astype(np.float32),                       # shape (Ntimes,)
        freq=freq.astype(np.float32),                       # shape (Nfreqs,)
        rfi_mask=rfi_mask,                                  # masked channels, shape (Nfreqs,)
        RMSF=RMSF.astype(np.complex64),                     # RMSF w/ masked channels, shape (Nphi,)
        RMSF_full=RMSF_full.astype(np.complex64),           # RMSF w/out masked channels, shape (Nphi,)
        FDF_arr=FDF_arr.astype(np.complex64),               # shape (Ntimes_fdf, Nphi)
        RM_meas_arr=RM_meas_arr.astype(np.float32),         # phi where peak of FDF occurs, for each time slice. shape (Ntimes_fdf,)
        time_slice_arr=time_slice_arr.astype(np.float32),   # shape (Ntimes_fdf,)
        lambda2_array=lambda2_array.astype(np.float32),     # shape (Nlambda,) = (Nfreq,)
        phi_array=phi_array.astype(np.float32),             # shape (Nphi,)
        cross_corr_arr=cross_corr_arr.astype(np.float32),   # shape (Ntimes_fdf, Nphilags)
        phi_lags=phi_lags.astype(np.float32)                # shape (Nphilags,)
    )

    # -- Save metadata --
    np.savez_compressed(
        metadata_file,
        metadata=np.array([metadata], dtype=object)
    )

    # -- Save timing results --
    np.savez_compressed(
        timing_file,
        timings=np.array([TIMINGS], dtype=object)
    )


    # -- Save summary text file --
    with open(summary_file, "w") as SUMMARY_FILE:
        print("=" * 80, file=SUMMARY_FILE)
        print("RM SEARCH SUMMARY", file=SUMMARY_FILE)
        print("=" * 80, file=SUMMARY_FILE)

        # Input / output
        print("\n[FILES]", file=SUMMARY_FILE)
        print(f"Source directory : {SOURCE_DIR}", file=SUMMARY_FILE)
        print(f"Observation night: {OBS_NIGHT}", file=SUMMARY_FILE)
        print(f"Arrays file      : {arrays_file}", file=SUMMARY_FILE)
        print(f"Metadata file    : {metadata_file}", file=SUMMARY_FILE)
        print(f"Timing file      : {timing_file}", file=SUMMARY_FILE)

        # Data info
        print("\n[DATA]", file=SUMMARY_FILE)
        print(f"Read {NFILES} files ({NUMBER_OF_FRAMES} frames)", file=SUMMARY_FILE)
        print(f"Stokes shape           : {stokes_norm_masked.shape} (Nstokes, Ntimes, Nfreqs)", file=SUMMARY_FILE)
        print(f"Frequency range        : {freq[0]:.2f} - {freq[-1]:.2f} MHz", file=SUMMARY_FILE)
        print(f"Time duration          : {time[-1] - time[0]:.3f} s", file=SUMMARY_FILE)
        print(f"Time resolution (dt)   : {dt_stokes:.4f}", file=SUMMARY_FILE)
        print(f"Frequency resolution   : {df_stokes:.4f}", file=SUMMARY_FILE)

        # RM search info
        print("\n[RM SEARCH]", file=SUMMARY_FILE)
        print(f"phi_max                : {PHI_MAX:.3f} rad/m^2", file=SUMMARY_FILE)
        print(f"dphi                   : {dphi:.5f} rad/m^2", file=SUMMARY_FILE)
        print(f"RMSF FWHM              : {fwhm:.5f} rad/m^2", file=SUMMARY_FILE)
        print(f"Nphi                   : {len(phi_array)}", file=SUMMARY_FILE)
        print(f"Search time step       : {SEARCH_TIME_STEP}", file=SUMMARY_FILE)
        print(f"New Ntime for RM search    : {len(time_slice_arr)}", file=SUMMARY_FILE)
        print(f"Effective dt for RM search : {dt_stokes * SEARCH_TIME_STEP:.4f}", file=SUMMARY_FILE)

        # RFI info
        print("\n[RFI MASKING]", file=SUMMARY_FILE)
        print(f"Num of masked channels : {np.sum(rfi_mask)} / {len(rfi_mask)}", file=SUMMARY_FILE)
        print(f"Masked fraction        : {100*np.sum(rfi_mask)/len(rfi_mask):.2f} %", file=SUMMARY_FILE)
        print(f"Mean threshold         : {RFI_MEAN_THRESHOLD}", file=SUMMARY_FILE)
        print(f"Std threshold          : {RFI_STD_THRESHOLD}", file=SUMMARY_FILE)

        # Simulation info
        print("\n[SIMULATION]", file=SUMMARY_FILE)
        print(f"SIM_FLAG        : {SIM_FLAG}", file=SUMMARY_FILE)
        if SIM_FLAG:
            print(f"Simulated {SIM_NUM_BURSTS} bursts", file=SUMMARY_FILE)
            print(f"Burst RMs   : {SIM_RM}", file=SUMMARY_FILE)
            print(f"Bust S/Ns   : {SIM_SNR}", file=SUMMARY_FILE)

        # Timing info
        print("\n[TIMINGS]", file=SUMMARY_FILE)
        for block, vals in TIMINGS.items():
            print(
                f"{block:25s} | "
                f"wall = {vals['wall_elapsed']:.3f} s | "
                f"cpu = {vals['cpu_elapsed']:.3f} s | "
                f"eff = {vals['cpu_efficiency']:.3f}",
                file=SUMMARY_FILE
            )

            if vals["mem_peak_gb"] is not None:
                print(
                    f"{'':25s} | "
                    f"peak mem = {vals['mem_peak_gb']:.3f} GB",
                    file=SUMMARY_FILE
                )

        print("\n[DONE]", file=SUMMARY_FILE)
        print(f"Saved RM search arrays to {arrays_file}", file=SUMMARY_FILE)
        print(f"Saved metadata to {metadata_file}", file=SUMMARY_FILE)
        print(f"Saved timing results to {timing_file}", file=SUMMARY_FILE)

    print(f"Saved summary to {summary_file}")
    print(f"Saved arrays to {arrays_file}")
    print(f"Saved metadata to {metadata_file}")
    print(f"Saved timing results to {timing_file}")