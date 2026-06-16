"""
Script for plotting *all* results of RM search when varying a single parameter. 

"""

# Imports
# ------------------------------------------------------------------------------
from plot_utils import *  # contains all plotting functions
from pulse_utils import *  # contains functions for finding/counting pulses in stokes and FDF
import numpy as np
import astropy.units as u
import glob
import os
import argparse


# Get arguments
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("results_dir", type=str,
                    help='Directory where search results are stored.')
parser.add_argument("settings", type=str, 
                    help='Specify which Stokes data was used in the search.')
parser.add_argument("param", type=str, 
                    help='Specify the varied parameter to plot results for.')
parser.add_argument("source_P0", type=float,
                    help='Pulse period of the source in seconds.')
parser.add_argument("source_W50", type=float,
                    help='Pulse width (W50) of the source in milliseconds.')
parser.add_argument("tol", type=float,
                    help='Minimum time separation (in ms) between detected peaks in FDF to be counted as separate pulses.')

parser.add_argument("--pulse_search_downsamp_factor", type=int, default=8,
                    help='Downsampling factor to apply to FDF_arr before finding peaks for pulse detection.')
parser.add_argument("--prominence_factor", type=float, default=10,
                    help='Prominence cutoff factor to apply when finding peaks in FDF for pulse detection.')
parser.add_argument("--param_label", type=str, default=None,
                    help='Label for the x-axis corresponding to the varied parameter (e.g., "Downsampling Factor", "DM [pc/cm^3]").')
parser.add_argument("--burst_lims", nargs='+', type=float, default=None,
                    help='Time limits (in seconds) to average over burst in some plots. Provide as two numbers: --burst_lims tmin tmax')
parser.add_argument("--data_files_unique", type=int, default=1,
                    help='Flag to indicate whether the masked+normalized Stokes data is ' \
                    'the same for all values of the varied parameter (1) or if it changes ' \
                    'with each run (0).')
parser.add_argument("--plot_fdf", type=int, default=1,
                    help='Flag to indicate whether to plot FDFs and stokes data (1) or not (0).')

args = parser.parse_args()

# Info to get search results
RESULTS_DIR = args.results_dir
SETTINGS = args.settings
PARAM = args.param
PARAM_LABEL = args.param_label if args.param_label is not None else PARAM  # use provided label or default to param name
PARAM_DIR = f"{RESULTS_DIR}/{SETTINGS}/{PARAM}/"
BURST_LIMS = args.burst_lims
if BURST_LIMS == [0]:
    BURST_LIMS = None  # interpret burst_lims of 0 as no limits (i.e. use full time range for plots)

# Search result files
METADATA_FILES = sorted(glob.glob(os.path.join(PARAM_DIR, f"{PARAM}*_metadata.npz")), key=len)
TIMINGS_FILES = sorted(glob.glob(os.path.join(PARAM_DIR, f"{PARAM}*_timings.npz")), key=len)
DATA_FILES = sorted(glob.glob(os.path.join(PARAM_DIR, f"{PARAM}*_arrays.npz")), key=len)
NFILES = len(DATA_FILES)  # number of files to plot
DATA_FILES_UNIQUE = args.data_files_unique   # determines if masked+normalized stokes data is the same for all values of the varied params or not
                                             # True if data is unique, false if data changes with each run of param

# Pulse info
SOURCE_P0 = args.source_P0 * u.s  # pulse period
SOURCE_W50 = args.source_W50 * u.ms  # pulse width (W50)
PULSE_SEARCH_DOWNSAMP_FACTOR = args.pulse_search_downsamp_factor  # downsampling factor to apply to FDF_arr before finding peaks
PROMINENCE_FACTOR = args.prominence_factor  # Prominence cutoff to apply when finding peaks in FDF for pulse detection
TOL = args.tol * u.ms  # min distance bw peaks for them to be counted as 2 peaks (in ms)

# Save directory for figures
SOURCE_AND_NIGHT = f"{RESULTS_DIR.split('/')[-2]}/{RESULTS_DIR.split('/')[-1]}"
SAVE_FIG_DIR = f'/scratch/abernier/results_figures/{SOURCE_AND_NIGHT}/{SETTINGS}/{PARAM}/'
os.makedirs(os.path.dirname(SAVE_FIG_DIR), exist_ok=True)
SAVE_BURST_DIR = SAVE_FIG_DIR + f'/burst_detected_downsamp{PULSE_SEARCH_DOWNSAMP_FACTOR}_tol{TOL.value}/'  # subdirectory for burst detection plots
os.makedirs(SAVE_BURST_DIR, exist_ok=True)
os.makedirs(SAVE_FIG_DIR+'/FDFs/', exist_ok=True)  # subdirectory for fdf plots
os.makedirs(SAVE_FIG_DIR+'/stokes/', exist_ok=True)  # subdirectory for stokes plots
os.makedirs(SAVE_FIG_DIR+'/visible_stokesI/', exist_ok=True)  # subdirectory for stokes I plots with visible pulses marked
SUMMARY_FILE = f"{RESULTS_DIR}/{SETTINGS}/{PARAM}_plot_summary_downsamp{PULSE_SEARCH_DOWNSAMP_FACTOR}_tol{TOL.value}.txt"



if __name__ == "__main__":
    # Open Results from Files
    # ------------------------------------------------------------------------------
    param_list = []
    data = {}
    metadata = {}
    timings = {}

    for data_file, metadata_file, timings_file in zip(DATA_FILES, METADATA_FILES, TIMINGS_FILES):
        # Get parameter value from filename
        filename = os.path.basename(data_file)  # format: param_name_{param_value}_arrays.npz
        param_value = float(filename.split("_")[-2]) 
        param_list.append(param_value)

        # Load data
        with np.load(data_file, allow_pickle=True) as df:
            data_dict = {k: df[k] for k in df.files}
            data[param_value] = data_dict

        # Load metadata
        with np.load(metadata_file, allow_pickle=True) as mf:
            metadata[param_value] = mf['metadata'][0]

        # Load timings
        with np.load(timings_file, allow_pickle=True) as tf:
            timings[param_value] = tf['timings'][0]
    
    param_list = sorted(param_list)  # sort parameter values in ascending order (just in case)
                                     # DATA_FILES is only sorted by length so eg. 141.26 gets put above 200

    # Get true RMs (using 1st file)
    rm_true = list(metadata[param_list[0]]['true_rms_rad_m2'].value)  # true RM of real data
    if metadata[param_list[0]]['sim_flag']:
        sim_rm = metadata[param_list[0]]['sim_params']["rm"]  # true RM of injected bursts
        rm_true.append(sim_rm.value)
    rm_true = np.array(rm_true)  # shape (Ntrue_rms,)


    # Print info to summary file (overwrite)
    # ------------------------------------------------------------------------------
    with open(SUMMARY_FILE, "w") as summary_file:
        # Header
        print("=" * 80, file=summary_file)
        print("RM SEARCH PLOT SUMMARY", file=summary_file)
        print("=" * 80, file=summary_file)

        # General info
        print("\n[RUN & SOURCE INFO]", file=summary_file)
        print(f"Results directory : {RESULTS_DIR}", file=summary_file)
        print(f"Settings          : {SETTINGS}", file=summary_file)
        print(f"Varied parameter  : {PARAM}", file=summary_file)
        print(f"Number of runs    : {NFILES}", file=summary_file)
        print(f"Parameter values  : {sorted(param_list)}", file=summary_file)

        # Metadata contents
        first_param = sorted(param_list)[0]
        print("\n[METADATA CONTENTS]", file=summary_file)
        print("Stored per-run search settings and results.", file=summary_file)
        for k in metadata[first_param].keys():
            print(f"  - {k}", file=summary_file)

        # Timing contents
        print("\n[TIMING CONTENTS]", file=summary_file)
        print("Stored runtime diagnostics for each processing stage.", file=summary_file)
        for k in timings[first_param].keys():
            print(f"  - {k}", file=summary_file)

        # Array contents
        print("\n[DATA ARRAYS]", file=summary_file)
        print(f"Example of arrays available for plotting (using {PARAM}={first_param}):", file=summary_file)
        for k, arr in data[first_param].items():
            try:
                print(
                    f"  - {k:<25} "
                    f"shape={str(arr.shape):<18} "
                    f"dtype={arr.dtype}",
                    file=summary_file
                )
            except Exception:
                print(f"  - {k}", file=summary_file)
        
        # Pulse detection parameters
        print("\n[PULSE FINDING]", file=summary_file)
        print(f"Pulse period (P0) : {SOURCE_P0}", file=summary_file)
        print(f"Pulse width (W50) : {SOURCE_W50}", file=summary_file)
        print(f'Tolerance to differentiate pulses: {TOL}', file=summary_file)
        print(f'Prominence cutoff : {PROMINENCE_FACTOR}', file=summary_file)


    # Plot Stokes & RMSF
    # ------------------------------------------------------------------------------
    if args.plot_fdf:
        for i,p in enumerate(param_list):
            # Get file labels
            if DATA_FILES_UNIQUE:
                label_stokes = 'stokes_data'
                label_rmsf = 'RMSF'
            else:
                label_stokes = f'stokes_data_{PARAM}_{p}'
                label_rmsf = f'RMSF_{PARAM}_{p}'
                
            # Plot masked + normalized Stokes data
            stokes_norm_masked = data[p]['full_stokes']  # shape (Nstokes, Ntimes, Nfreqs)
            freq = data[p]['freq']  # shape (Nfreqs,)
            time = data[p]['time']  # shape (Ntimes,)
            plot_stokes(stokes_norm_masked, freq=freq, time=time, cbar_lim_max=None,
                        t_unit='s', save_name=label_stokes, save_loc=SAVE_FIG_DIR+'/stokes/')

            # Plot RMSF
            # RMSF = data[p]['RMSF']  # shape (Nphi,)
            # RMSF_full = data[p]['RMSF_full']  # shape (Nphi,)
            # phi_array = data[p]['phi_array']  # shape (Nphi,)
            # plot_rmsf(phi_array, RMSF, RMSF_full, save_name=label_rmsf, save_loc=SAVE_FIG_DIR)

            if DATA_FILES_UNIQUE:  # if data is the same for all runs, only plot once
                break


    # Find pulses + Plot FDF and cross-correlation results
    # ------------------------------------------------------------------------------
    # Setup counters to keep track of pulse detection stats across all runs
    num_pulses_detected_tot = 0  # through FDF
    num_pulses_visible_tot = 0  # estimated through stokes I
    num_pulses_found_notvisible_tot = 0  # pulses found in FDF that don't appear visible in stokes I 
    total_time_length = 0  # total time length of all data read (to estimate total number of expected pulses across all runs)

    for p in param_list:
        # -- Get data arrays --
        stokes_arr = data[p]['full_stokes']  # shape (Nstokes, Ntimes, Nfreqs)
        time_arr = data[p]['time']  # shape (Ntimes,)
        FDF_arr = data[p]['FDF_arr']  # shape (Ntime_fdf, Nphi)
        time_slice_arr = data[p]['time_slice_arr']  # shape (Ntime_fdf,)
        dt_fdf = metadata[p]['dt_rm_ms']  # time resolution of FDF_arr
        time_length = time_slice_arr[-1]*u.s - time_slice_arr[0]*u.s
        phi_array = data[p]['phi_array']  # shape (Nphi,)
        cross_corr_arr = data[p]['cross_corr_arr']  # shape (Ntime_fdf, Nphi_lags)
        phi_lags = data[p]['phi_lags']  # shape (Nphi_lags,)

        # -- Count number of visible pulses (using stokes I) --
        num_pulses_visible, visible_groups = count_visible_pulses(stokes_arr, 
                                                                  time_arr, 
                                                                  max_dist=500,
                                                                  save_loc=SAVE_FIG_DIR+ '/visible_stokesI/',
                                                                  save_name=f'vis_{PARAM}_{p}.png'
                                                                  )  # see fct doc for output format
        num_pulses_visible_tot += num_pulses_visible
        total_time_length += time_length

        # -- Find pulses in FDF (& plot them) --
        detected_bursts = find_pulses(p, FDF_arr, phi_array, time_slice_arr, rm_true=rm_true,
                                      downsamp_factor=PULSE_SEARCH_DOWNSAMP_FACTOR, 
                                      time_tol=TOL, prominence_factor=PROMINENCE_FACTOR,
                                      save_loc=SAVE_BURST_DIR
                                      )  # see fct doc for output format
        num_pulses_detected_tot += len(detected_bursts)

        # -- Count how many detected bursts don't appear visible in stokes I --
        num_found_notvisible = count_found_notvisible(detected_bursts, visible_groups, time_arr)
        num_pulses_found_notvisible_tot += num_found_notvisible

        # -- Print pulse detection info to summary file --
        with open(SUMMARY_FILE, "a") as summary_file:
            print(f"\nProcessing {PARAM}={p}...", file=summary_file)
            print(f'    Downsampling to {dt_fdf*PULSE_SEARCH_DOWNSAMP_FACTOR:.3f} ms', file=summary_file)
            print(f"    FDF time length: {time_length:.3f}", file=summary_file)
            print(f"    Expecting ~{int(time_length / SOURCE_P0)} pulses in data", file=summary_file)
            print(f"    Estimated number of visible pulses in Stokes I: {num_pulses_visible}", file=summary_file)
            print(f"    Found {len(detected_bursts)} pulse(s) in the FDF", file=summary_file)
            print(f"    Found {num_found_notvisible} pulse(s) in FDF that don't appear visible in Stokes I", file=summary_file)

        
        if args.plot_fdf:
            # FDF - imshow
            plot_2panels(FDF_arr, time_slice_arr, phi_array, 
                        cbar_label='Amplitude', suptitle='FDF', 
                        save_name=f'FDF_{PARAM}_{p}', save_loc=SAVE_FIG_DIR)
        
            # Cross-correlation - imshow
            # plot_2panels(cross_corr_arr, time_slice_arr, phi_lags, ylim=(-1500,1500),
            #              cbar_label='Amplitude', suptitle='Cross-correlation',
            #              save_name=f'crosscorr_{PARAM}_{p}', save_loc=SAVE_FIG_DIR)
        
            # Cross-correlation - slices plot
            # plot_cross_corr_slices(phi_lags, cross_corr_arr, time,
            #                        true_RMs=rm_true,
            #                        save_name=f'crosscorr_slices_{PARAM}_{p}',
            #                        save_loc=SAVE_FIG_DIR
            #                        )
    

    # Print statistics about pulse detection across all runs to summary file
    # ------------------------------------------------------------------------------
    num_bursts_expected_tot = int(total_time_length / SOURCE_P0)  # based on time length of data read and pulse period
    num_pulses_not_visible_tot = num_bursts_expected_tot - num_pulses_visible_tot  # pulses that appear in FDF but don't appear visible in stokes I
    
    with open(SUMMARY_FILE, "a") as summary_file:
        print("\n[OVERALL PULSE DETECTION SUMMARY]", file=summary_file)
        print(f"Total number of pulses expected across all runs (based on time length and pulse period): {num_bursts_expected_tot}", file=summary_file)
        print(f"Total number of visible pulses across all runs (estimated from Stokes I): {num_pulses_visible_tot}", file=summary_file)
        print(f"Total number of pulses not visible in Stokes I (expected - visible): {num_pulses_not_visible_tot}", file=summary_file)
        print(f"Total number of pulses detected across all runs (using FDF): {num_pulses_detected_tot}", file=summary_file)
        print(f"Total number of pulses found in FDF that don't appear visible in Stokes I: {num_pulses_found_notvisible_tot}", file=summary_file)

   

    # Time Curves
    # ------------------------------------------------------------------------------
    if args.plot_fdf:
        # wall & cpu times vs param, grouped by block
        plot_timings_by_block(param_list, metadata, timings, param_name=PARAM, param_label=PARAM_LABEL,
                            fig_title=None, plot_type='cpu_elapsed', save_name=f'timecurves', save_loc=SAVE_FIG_DIR,
                            plot_total=True, plot_wall=False, plot_ylog=True, convert_tstep=False)
        
        # cpu efficiency vs param, grouped by code block 
        plot_timings_by_block(param_list, metadata, timings, param_name=PARAM, param_label=PARAM_LABEL,
                            fig_title=None, plot_type='cpu_efficiency', save_name=f'efficiency', save_loc=SAVE_FIG_DIR,
                            plot_total=True, plot_ylog=False, convert_tstep=False)