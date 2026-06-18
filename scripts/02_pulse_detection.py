"""
Script to search for pulses in FDFs (obtained from RM search results
generated using different values of a varied parameter). 
"""


# Imports
# ==============================================================================
from pulse_utils import *  # functions for finding/counting pulses in stokes and FDF
import numpy as np
import astropy.units as u
import glob
import os
import argparse
from collections import defaultdict


# Parse arguments
# ==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("results_dir", type=str,
                    help='Absolute path to directory containing search results.')
parser.add_argument("save_fig_dir", type=str, 
                    help='Absolute path to directory where pulse detection results should be saved.')
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
parser.add_argument("--plot_detected_pulses", type=int, default=0,
                    help='Whether to save plots of detected pulses in FDF for each parameter value (1 to save, 0 to not save).')
args = parser.parse_args()

# Paths
RESULTS_DIR = args.results_dir
SAVE_FIG_DIR = args.save_fig_dir

# Source properties
SOURCE_P0 = args.source_P0 * u.s  # pulse period
SOURCE_W50 = args.source_W50 * u.ms  # pulse width (W50)

# Pulse-finding parameters
PULSE_SEARCH_DOWNSAMP_FACTOR = args.pulse_search_downsamp_factor  # downsampling factor to apply to FDF_arr before finding peaks
PROMINENCE_FACTOR = args.prominence_factor  # Prominence cutoff to apply when finding peaks in FDF for pulse detection
TOL = args.tol * u.ms  # min distance bw peaks for them to be counted as 2 peaks (in ms)

# Flags
PLOT_DETECTED_PULSES = args.plot_detected_pulses 


# Get RM search result files
# ==============================================================================
METADATA_FILES = sorted(
    glob.glob(os.path.join(RESULTS_DIR, "*_metadata.npz")),
    key=len
)
TIMINGS_FILES = sorted(
    glob.glob(os.path.join(RESULTS_DIR, "*_timings.npz")),
    key=len
)
DATA_FILES = sorted(
    glob.glob(os.path.join(RESULTS_DIR, "*_arrays.npz")),
    key=len
)
NFILES = len(DATA_FILES)


# Define output directories and files
# ==============================================================================
BURST_VIS_RESULTS = (       # file to save pulses visible in stokes I
    f"{SAVE_FIG_DIR}/burst_visible_info.npz"
)

if PLOT_DETECTED_PULSES:
    BURST_DETECT_DIR = (        # subdirectory for burst detection plots
        SAVE_FIG_DIR
        + f"/tol{TOL.value:.0f}/burst_detected_downsamp"
        f"{PULSE_SEARCH_DOWNSAMP_FACTOR}"
        f"_tol{TOL.value:.0f}/"
    )
    os.makedirs(BURST_DETECT_DIR, exist_ok=True)
else:
    BURST_DETECT_DIR = None

BURST_DETECT_RESULTS = (    # file to save detected bursts info
    f"{SAVE_FIG_DIR}/tol{TOL.value:.0f}/burst_detected_info_downsamp"
    f"{PULSE_SEARCH_DOWNSAMP_FACTOR}"
    f"_tol{TOL.value:.0f}.npz"
)

SUMMARY_FILE = (
    f"{SAVE_FIG_DIR}/tol{TOL.value:.0f}/"
    f"summary_downsamp"
    f"{PULSE_SEARCH_DOWNSAMP_FACTOR}"
    f"_tol{TOL.value:.0f}.txt"
)


# Load or initialize visible burst results
# ==============================================================================
if os.path.isfile(BURST_VIS_RESULTS):  # if file already exists, read it
    visible_dict = np.load(
        BURST_VIS_RESULTS,
        allow_pickle=True
    )['visible_dict'].item()
else:
    visible_dict = {}  # if files doesn't exist, create empty dict to be filled
  

# Main script
# ==============================================================================
if __name__ == "__main__":

    # Load RM search results
    # --------------------------------------------------------------------------
    param_list = []
    data = {}
    metadata = {}

    for data_file, metadata_file in zip(DATA_FILES, METADATA_FILES):
        # Get parameter value from filename
        filename = os.path.basename(data_file)  # format: param_name_{param_value}_arrays.npz
        param_value = float(filename.split("_")[-2]) 
        param_list.append(param_value)

        # Load data arrays
        with np.load(data_file, allow_pickle=True) as df:
            data_dict = {k: df[k] for k in df.files}
            data[param_value] = data_dict

        # Load metadata
        with np.load(metadata_file, allow_pickle=True) as mf:
            metadata[param_value] = mf['metadata'][0]
    
    # sort parameter values in ascending order (just in case)
    # DATA_FILES is only sorted by length so eg. 141.26 gets put above 200
    param_list = sorted(param_list)
    
    # Load first timing file to get timing keys (assuming all files have same keys)
    with np.load(TIMINGS_FILES[0], allow_pickle=True) as tf:
            timings = tf['timings'][0]


    # Get true RMs (using 1st file)
    # --------------------------------------------------------------------------
    rm_true = list(metadata[param_list[0]]['true_rms_rad_m2'].value)  # true RM of real data
    if metadata[param_list[0]]['sim_flag']:
        sim_rm = metadata[param_list[0]]['sim_params']["rm"]  # true RM of injected bursts
        rm_true.append(sim_rm.value)
    rm_true = np.array(rm_true)  # shape (Ntrue_rms,)


    # Print info to summary file
    # --------------------------------------------------------------------------
    with open(SUMMARY_FILE, "w") as summary_file:
        # Header
        print("=" * 80, file=summary_file)
        print("PULSE DETECTION SUMMARY", file=summary_file)
        print("=" * 80, file=summary_file)

        # General info
        print("\n[RUN & SOURCE INFO]", file=summary_file)
        source_dir = RESULTS_DIR.split("/")[:-2]  # assuming path format .../results_data/{source}/{obs_night}/{settings}/{subdir}
        settings = RESULTS_DIR.split("/")[-2]
        subdir = RESULTS_DIR.split("/")[-1]
        print(f"Results directory : {source_dir}", file=summary_file)
        print(f"Settings          : {settings}", file=summary_file)
        print(f"Varied parameter  : {subdir}", file=summary_file)
        print(f"Number of runs    : {NFILES}", file=summary_file)
        print(f"Parameter values  : {param_list}", file=summary_file)

        # Metadata contents
        print("\n[METADATA CONTENTS]", file=summary_file)
        print("Stored per-run search settings and results.", file=summary_file)
        first_param = param_list[0]
        for k in metadata[first_param].keys():
            print(f"  - {k}", file=summary_file)

        # Timing contents
        print("\n[TIMING CONTENTS]", file=summary_file)
        print("Stored runtime diagnostics for each processing stage.", file=summary_file)
        for k in timings.keys():
            print(f"  - {k}", file=summary_file)

        # Array contents
        print("\n[DATA ARRAYS]", file=summary_file)
        print(f"Example of arrays available:", file=summary_file)
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


    # Find pulses
    # --------------------------------------------------------------------------
    # Counters to keep track of pulse detection stats across all runs
    num_pulses_visible_tot = 0  # estimated through stokes I
    num_pulses_detected_tot = 0  # through FDF
    num_pulses_detected_notvisible_tot = 0  # pulses detected in FDF that don't appear visible in stokes I 
    total_time_length = 0  # total time length of all data read (to estimate total number of expected pulses across all runs)

    # dict to store results related to detected pulses
    results = {
        "params": {},  # by parameter results
        "summary": {}  # overall summary stats across all runs
    }

    for p in param_list:

        # ----- Get data arrays -----
        stokes_arr = data[p]['full_stokes']  # shape (Nstokes, Ntimes, Nfreqs)
        time_arr = data[p]['time']  # shape (Ntimes,)
        FDF_arr = data[p]['FDF_arr']  # shape (Ntime_fdf, Nphi)
        time_slice_arr = data[p]['time_slice_arr']  # shape (Ntime_fdf,)
        dt_fdf = metadata[p]['dt_rm_ms']  # time resolution of FDF_arr
        phi_array = data[p]['phi_array']  # shape (Nphi,)

        # ----- Update total time -----
        time_length = time_slice_arr[-1]*u.s - time_slice_arr[0]*u.s
        total_time_length += time_length

        # ----- Count pulses visible in stokes I -----
        if p not in visible_dict: 
            # see fct doc for output format
            num_pulses_visible, visible_bursts = count_visible_pulses(
                stokes_arr, 
                time_arr, 
                max_dist=500  # number of time bins 
            )
            visible_dict[p] = {
                'num_pulses_visible': num_pulses_visible,
                'visible_bursts': visible_bursts,
            }   

        else:
            # Already computed -> get info from loaded file
            num_pulses_visible = visible_dict[p]['num_pulses_visible']
            visible_bursts = visible_dict[p]['visible_bursts']

        num_pulses_visible_tot += num_pulses_visible  # update total count of visible pulses across all runs

        # ----- Find pulses in FDF -----
        # see fct doc for output format
        num_pulses_detected, detected_bursts = find_pulses_fdf(
            FDF_arr, phi_array, time_slice_arr, rm_true=rm_true,
            downsamp_factor=PULSE_SEARCH_DOWNSAMP_FACTOR, 
            time_tol=TOL, 
            prominence_factor=PROMINENCE_FACTOR,
            height=None,
            save_loc=BURST_DETECT_DIR,
            save_name=f'detect_param_{p:.0f}.png'
        ) 
        num_pulses_detected_tot += num_pulses_detected

        # ----- Count how many detected bursts don't appear visible in stokes I -----
        num_detected_notvisible, notvisible_intervals = count_found_notvisible(
            detected_bursts, 
            visible_bursts, 
            time_arr
        )
        num_pulses_detected_notvisible_tot += num_detected_notvisible

        # ----- Print pulse detection info to summary file -----
        with open(SUMMARY_FILE, "a") as summary_file:
            print(f"\nProcessing param={p}...", file=summary_file)
            print(f'    Downsampling to {dt_fdf*PULSE_SEARCH_DOWNSAMP_FACTOR:.3f} ms', file=summary_file)
            print(f"    FDF time length: {time_length:.3f}", file=summary_file)
            print(f"    Expecting ~{int(time_length / SOURCE_P0)} pulses in data", file=summary_file)
            print(f"    Estimated number of visible pulses in Stokes I: {num_pulses_visible}", file=summary_file)
            print(f"    Found {len(detected_bursts)} pulse(s) in the FDF", file=summary_file)
            print(f"    Found {num_detected_notvisible} pulse(s) in FDF that don't appear visible in Stokes I", file=summary_file)
            print(
                "    Intervals of detected pulses that don't appear visible in Stokes I:",
                file=summary_file,
            )
            for start, end in notvisible_intervals:
                print(
                    f"      {start:.3f} s - {end:.3f} s",
                    file=summary_file,
                )
        
        # ----- Save results for this parameter value -----
        results["params"][p] = {
            "detected_bursts": detected_bursts,
            "notvisible_intervals": notvisible_intervals,
        }


    # Print statistics about pulse detection across all runs to summary file
    # ---------------------------------------------------------------------------
    num_bursts_expected_tot = int(total_time_length / SOURCE_P0)  # based on time length of data read and pulse period
    num_pulses_not_visible_tot = num_bursts_expected_tot - num_pulses_visible_tot  # pulses that appear in FDF but don't appear visible in stokes I
    
    with open(SUMMARY_FILE, "a") as summary_file:
        print("\n[OVERALL PULSE DETECTION SUMMARY]", file=summary_file)
        print(f"Total number of pulses expected across all runs (based on time length and pulse period): {num_bursts_expected_tot}", file=summary_file)
        print(f"Total number of visible pulses across all runs (estimated from Stokes I): {num_pulses_visible_tot}", file=summary_file)
        print(f"Total number of pulses not visible in Stokes I (expected - visible): {num_pulses_not_visible_tot}", file=summary_file)
        print(f"Total number of pulses detected across all runs (using FDF): {num_pulses_detected_tot}", file=summary_file)
        print(f"Total number of pulses found in FDF that don't appear visible in Stokes I: {num_pulses_detected_notvisible_tot}", file=summary_file)


    # Add to results the overall stats across all runs
    # -----------------------------------------------------------------------------
    results["summary"] = {
        "num_bursts_expected_tot": num_bursts_expected_tot,
        "num_pulses_visible_tot": num_pulses_visible_tot,
        "num_pulses_not_visible_tot": num_pulses_not_visible_tot,
        "num_pulses_detected_tot": num_pulses_detected_tot,
        "num_pulses_detected_notvisible_tot": num_pulses_detected_notvisible_tot,
    }


    # Save to file
    # --------------------------------------------------------------------------
    # Visible pulse info (only for first run)
    if PULSE_SEARCH_DOWNSAMP_FACTOR == 4: 
        np.savez(BURST_VIS_RESULTS, visible_dict=visible_dict)

    # Detected pulses
    np.savez(BURST_DETECT_RESULTS, results=results)


# RESULTS FORMAT
# =============================================================================

# results = {
#     "params": {
#         p: {
#          "detected_bursts": [list of detected bursts in FDF for this param value],
#               keys of inner dict: 't_ind' (time index of FDF slice), 
#                                   'time' (time of FDF slice), 
#                                   'peaks' (list of indices of peaks in FDF, along phi axis), 
#                                   'median' (median value of FDF for this time slice)
#               Eg. detected_bursts = [  
#                   [ {'t_ind': 0, 'time': 0.1, 'peaks': [10, 50], 'median': 5}, {'t_ind': 1, 'time': 0.2, 'peaks': [12], 'median': 4} ],  # burst with 2 detections
#                   [ {'t_ind': 10, 'time': 1.0, 'peaks': [30], 'median': 3} ]   # burst with 1 detection
#               ]
#          "notvisible_intervals": [list of intervals of detected bursts that don't appear visible in stokes I for this param value]
#               Eg. [(start1, end1), (start2, end2), ...]
#         },
#        ...
#    },
#    "summary": {
#        "num_bursts_expected_tot": int,  # based on time length of data read and pulse period
#        "num_pulses_visible_tot": int,  # estimated through stokes I
#        "num_pulses_not_visible_tot": int,  # expected - visible
#        "num_pulses_detected_tot": int,  # detected in FDF
#        "num_pulses_detected_notvisible_tot": int,  # pulses detected in FDF that don't appear visible in stokes I
#    }
# }

# Eg. To get all the detections for the nth detected burst for the mth parameter value:
# param_value = param_list[m]
# detected_bursts = results["params"][param_value]["detected_bursts"]
# nth_burst = detected_bursts[n]
# Each element of nth_burst is a dict with keys: 't_ind', 'time', 'peaks', 'median', 
# where nth_burst[i]['peaks'] gives the phi indices of the detected peaks


# Visible dict format
# ==============================================================================
# visible_dict = {
#     param_value1: {
#         'num_pulses_visible': int,
#         'visible_bursts': [list of pulse indices (in time) visible in stokes I for this param value]
#     },
#     param_value2: {
#         ...
#     }
#     ...
# }