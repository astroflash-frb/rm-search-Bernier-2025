"""
Script for plotting *all* results of RM search when varying a single parameter. 

"""

# Imports
# ------------------------------------------------------------------------------
from plot_utils import *  # contains all plotting functions
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

parser.add_argument("--pulse_search_downsamp_factor", type=int, default=8,
                    help='Downsampling factor to apply to FDF_arr before finding peaks for pulse detection.')
parser.add_argument("--snr_cutoff", type=float, default=10,
                    help='S/N cutoff to apply when finding peaks in FDF for pulse detection.')
parser.add_argument("--param_label", type=str, default=None,
                    help='Label for the x-axis corresponding to the varied parameter (e.g., "Downsampling Factor", "DM [pc/cm^3]").')
parser.add_argument("--burst_lims", nargs='+', type=float, default=None,
                    help='Time limits (in seconds) to average over burst in some plots. Provide as two numbers: --burst_lims tmin tmax')
parser.add_argument("--data_files_unique", type=int, default=1,
                    help='Flag to indicate whether the masked+normalized Stokes data is ' \
                    'the same for all values of the varied parameter (1) or if it changes ' \
                    'with each run (0).')

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

# Save directory for figures
SOURCE_AND_NIGHT = f"{RESULTS_DIR.split('/')[-2]}/{RESULTS_DIR.split('/')[-1]}"
SAVE_FIG_DIR = f'/scratch/abernier/rm-search-Bernier-2025/figures/{SOURCE_AND_NIGHT}/{SETTINGS}/{PARAM}/'
os.makedirs(os.path.dirname(SAVE_FIG_DIR), exist_ok=True)
os.makedirs(SAVE_FIG_DIR+'/burst_detected/', exist_ok=True)  # subdirectory for burst detection plots

SUMMARY_FILE = f"{RESULTS_DIR}/{SETTINGS}/{PARAM}_plot_summary.txt"

# Pulse info
SOURCE_P0 = args.source_P0 * u.s  # pulse period
SOURCE_W50 = args.source_W50 * u.ms  # pulse width (W50)
TOL = SOURCE_W50 + 10*u.ms  # min distance bw peaks for them to be counted as 2 peaks (in ms)
PULSE_SEARCH_DOWNSAMP_FACTOR = args.pulse_search_downsamp_factor  # downsampling factor to apply to FDF_arr before finding peaks
SNR_CUTOFF = args.snr_cutoff  # S/N cutoff to apply when finding peaks in FDF for pulse detection



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
        print(f'S/N cutoff        : {SNR_CUTOFF}', file=summary_file)



    # Plot Stokes & RMSF
    # ------------------------------------------------------------------------------
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
                    t_unit='s', save_name=label_stokes, save_loc=SAVE_FIG_DIR)

        # Plot RMSF
        # RMSF = data[p]['RMSF']  # shape (Nphi,)
        # RMSF_full = data[p]['RMSF_full']  # shape (Nphi,)
        # phi_array = data[p]['phi_array']  # shape (Nphi,)
        # plot_rmsf(phi_array, RMSF, RMSF_full, save_name=label_rmsf, save_loc=SAVE_FIG_DIR)

        if DATA_FILES_UNIQUE:  # if data is the same for all runs, only plot once
            break


    # Find pulses + Plot FDF and cross-correlation results
    # ------------------------------------------------------------------------------
    for p in param_list:
        # Get arrays to plot
        FDF_arr = data[p]['FDF_arr']  # shape (Ntime_fdf, Nphi)
        time_slice_arr = data[p]['time_slice_arr']  # shape (Ntime_fdf,)
        dt_fdf = metadata[p]['dt_rm_ms']  # time resolution of FDF_arr
        time_length = time_slice_arr[-1]*u.s - time_slice_arr[0]*u.s
        phi_array = data[p]['phi_array']  # shape (Nphi,)
        cross_corr_arr = data[p]['cross_corr_arr']  # shape (Ntime_fdf, Nphi_lags)
        phi_lags = data[p]['phi_lags']  # shape (Nphi_lags,)

        # Find pulses in FDF (& plot them)
        bursts = find_pulses(p, FDF_arr, phi_array, time_slice_arr, rm_true=rm_true,
                             downsamp_factor=PULSE_SEARCH_DOWNSAMP_FACTOR, 
                             time_tol=TOL, snr_cutoff=SNR_CUTOFF,
                             save_loc=SAVE_FIG_DIR+'/burst_detected/'
                             )
        
        # Add pulse detection info to data dict for this param value
        data[p]['bursts'] = np.array(bursts, dtype=object)  # add bursts to data dict for this param value
        data[p]['num_bursts_detected'] = len(bursts)  # store number of bursts found for this param value
        data[p]['pulse_search_downsamp_factor'] = PULSE_SEARCH_DOWNSAMP_FACTOR  # store downsamp factor used for pulse finding
        data[p]['snr_cutoff'] = SNR_CUTOFF  # store S/N cutoff used for pulse finding
        data[p]['pulse_separation_tol'] = TOL  # store time tolerance used to separate pulses
        
        # Print pulse detection info to summary file
        with open(SUMMARY_FILE, "a") as summary_file:
            print(f"\nProcessing {PARAM}={p}...", file=summary_file)
            print(f'    Downsampling to {dt_fdf*PULSE_SEARCH_DOWNSAMP_FACTOR:.3f} ms', file=summary_file)
            print(f"    FDF time length: {time_length:.3f}", file=summary_file)
            # expected pulses
            num_pulses_expected = time_length / SOURCE_P0  # num of expected pulses in data read
            print(f"    Expecting ~{int(num_pulses_expected)} pulses in data", file=summary_file)
            # pulses found
            print(f"    Found {len(bursts)} pulse(s)", file=summary_file)
        

        # FDF - imshow
        plot_2panels(FDF_arr, time_slice_arr, phi_array, 
                     cbar_label='Amplitude', suptitle='FDF', 
                     save_name=f'FDF_{PARAM}_{p}', save_loc=SAVE_FIG_DIR)
        
        # Cross-correlation - imshow
        # plot_2panels(cross_corr_arr, time_slice_arr, phi_lags, ylim=(-1500,1500),
        #              cbar_label='Amplitude', suptitle='Cross-correlation',
        #              save_name=f'crosscorr_{PARAM}_{p}', save_loc=SAVE_FIG_DIR)
    
        # Cross-correlation - slices plot
        # plot_cross_corr_slices(phi_lags, cross_corr_arr, 
        #                        true_RMs=rm_true,
        #                        save_name=f'crosscorr_slices_{PARAM}_{p}',
        #                        save_loc=SAVE_FIG_DIR
        #                        )
        

    # Time Curves
    # ------------------------------------------------------------------------------
    # wall & cpu times vs param, grouped by block
    plot_timings_by_block(param_list, metadata, timings, param_name=PARAM, param_label=PARAM_LABEL,
                          fig_title=None, plot_type='cpu_elapsed', save_name='timecurves', save_loc=SAVE_FIG_DIR,
                          plot_total=True, plot_wall=False, plot_ylog=True, convert_tstep=False)
    
    # cpu efficiency vs param, grouped by code block 
    plot_timings_by_block(param_list, metadata, timings, param_name=PARAM, param_label=PARAM_LABEL,
                          fig_title=None, plot_type='cpu_efficiency', save_name='efficiency', save_loc=SAVE_FIG_DIR,
                          plot_total=True, plot_ylog=False, convert_tstep=False)


    # Resave data dicts with added burst info
    # ------------------------------------------------------------------------------
    for p in param_list:
        data_file = f"{PARAM_DIR}/{PARAM}_{p}_arrays.npz"
        np.savez_compressed(data_file, **data[p])