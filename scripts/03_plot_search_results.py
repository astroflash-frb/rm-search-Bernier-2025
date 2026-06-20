"""
Script for plotting *all* results of RM search when varying a single parameter. 

"""


# Imports
# ==============================================================================
from plot_utils import *  # contains all plotting functions
import numpy as np
import astropy.units as u
import glob
import os
import argparse


# Parse arguments
# ==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("results_dir", type=str,
                    help='Absolute path to directory containing search results.')
parser.add_argument("save_fig_dir", type=str, 
                    help='Absolute path to directory where figures should be saved.')
parser.add_argument("tol", type=float,
                    help="Tolerance value to plot results for.")
parser.add_argument("--param_label", type=str, default=None,
                    help='Label for the x-axis corresponding to the varied parameter (e.g., "Downsampling Factor", "DM [pc/cm^3]").')
parser.add_argument("--data_files_unique", type=int, default=1,
                    help='Flag to indicate whether the masked+normalized Stokes data is ' \
                    'the same for all values of the varied parameter (1) or if it changes ' \
                    'with each run (0).')
args = parser.parse_args()

# Variables
RESULTS_DIR = args.results_dir
SAVE_FIG_DIR = args.save_fig_dir
TOL = args.tol
PARAM_LABEL = args.param_label if args.param_label is not None else "Varied Parameter"
DATA_FILES_UNIQUE = args.data_files_unique   # determines if masked+normalized stokes data is the same for all values of the varied params or not
                                             # True if data is unique, false if data changes with each run of param


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


# Get burst results files
# ==============================================================================
# file with info on pulses visible in stokes I
BURST_VIS_RESULTS = (
    f"{SAVE_FIG_DIR}/burst_visible_info.npz"
)  
if not os.path.isfile(BURST_VIS_RESULTS):
    print(f"Unable to find visible burst information at {BURST_VIS_RESULTS}. \n\
          Make sure to run 02_pulse_detection.py first to save visible pulse info.")
    exit(1)

# files with info on detected pulses in FDF
BURST_DETECT_FILES = sorted( 
    glob.glob(f"{SAVE_FIG_DIR}/tol{TOL:.0f}/burst_detected_info_downsamp*_tol{TOL:.0f}.npz"),
    key=len
)
if len(BURST_DETECT_FILES) == 0:
    print(f"Unable to find detected burst information at {BURST_DETECT_FILES}. \n\
          Make sure to run 02_pulse_detection.py first to save detected pulse info.")
    exit(1)


# Define output directories and files
# ==============================================================================
# subdirectory for stokes I timeseries plots with visible+detcted pulses marked
BURST_VIS_DIR = (
    f"{SAVE_FIG_DIR}/tol{TOL:.0f}/burst_visible_stokesI_tol{TOL:.0f}/"
)
os.makedirs(BURST_VIS_DIR, exist_ok=True)

# Stokes
stokes_dir = SAVE_FIG_DIR + '/stokes/'
stokes_exists = os.path.isdir(stokes_dir)  # check if Stokes plots already exist
os.makedirs(stokes_dir, exist_ok=True)

# FDFs
fdf_dir = SAVE_FIG_DIR + '/FDFs/'
fdf_exists = os.path.isdir(fdf_dir)  # check if FDF plots already exist
os.makedirs(fdf_dir, exist_ok=True)


# Load visible burst results
# ==============================================================================
visible_dict = np.load(
    BURST_VIS_RESULTS,
    allow_pickle=True
)['visible_dict'].item()


# Load detected burst results
# ==============================================================================
downsamp_list = []
results_params_all_downsamp = []
results_summary_all_downsamp = []

for file in BURST_DETECT_FILES:
    downsamp = int(os.path.basename(file).split("_")[-2][8:])  # assuming file name 'burst_detected_info_downsamp{FACTOR}_tol{TOL}.npz'
    downsamp_list.append(downsamp)
    
    results = np.load(file, allow_pickle=True)["results"].item()
    results_params_all_downsamp.append(results["params"])
    results_summary_all_downsamp.append(results["summary"])

  

# Main script
# ==============================================================================
if __name__ == "__main__":

    # Load RM search results
    # --------------------------------------------------------------------------
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
    
    # sort parameter values in ascending order (just in case)
    # DATA_FILES is only sorted by length so eg. 141.26 gets put above 200
    param_list = sorted(param_list)


    # Get true RMs (using 1st file)
    # --------------------------------------------------------------------------
    rm_true = list(metadata[param_list[0]]['true_rms_rad_m2'].value)  # true RM of real data
    if metadata[param_list[0]]['sim_flag']:
        sim_rm = metadata[param_list[0]]['sim_params']["rm"]  # true RM of injected bursts
        rm_true.append(sim_rm.value)
    rm_true = np.array(rm_true)  # shape (Ntrue_rms,)


    # Plot Stokes & RMSF
    # --------------------------------------------------------------------------
    if not stokes_exists:
        for i,p in enumerate(param_list):
            # Get file labels
            if DATA_FILES_UNIQUE:
                label_stokes = 'stokes'
                label_rmsf = 'RMSF'
            else:
                label_stokes = f'stokes_param_{p}'
                label_rmsf = f'RMSF_param_{p}'
                
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
            # plot_rmsf(phi_array, RMSF, RMSF_full, save_name=label_rmsf, save_loc=SAVE_FIG_DIR+'/RMSFs/')

            if DATA_FILES_UNIQUE:  # if data is the same for all runs, only plot once
                break
    else:
        print(f"Stokes plots already exist at {stokes_dir}. Skipping Stokes plotting step!")


    # Make plots by parameter value
    # --------------------------------------------------------------------------
    for p in param_list:

        # ----- Get data arrays -----
        stokes_arr = data[p]['full_stokes']  # shape (Nstokes, Ntimes, Nfreqs)
        time_arr = data[p]['time']  # shape (Ntimes,)
        FDF_arr = data[p]['FDF_arr']  # shape (Ntime_fdf, Nphi)
        time_slice_arr = data[p]['time_slice_arr']  # shape (Ntime_fdf,)
        phi_array = data[p]['phi_array']  # shape (Nphi,)
        # cross_corr_arr = data[p]['cross_corr_arr']  # shape (Ntime_fdf, Nphi_lags)
        # phi_lags = data[p]['phi_lags']  # shape (Nphi_lags,)

        # ----- FDF - imshow -----
        if not fdf_exists:
            plot_2panels(FDF_arr, time_slice_arr, phi_array, 
                        cbar_label='Amplitude', suptitle='FDF', 
                        save_name=f'FDF_param_{p}', save_loc=SAVE_FIG_DIR+"/FDFs/")
        else:
            print(f"FDF plots already exist at {fdf_dir}. Skipping FDF plotting step!")
    
        # ----- Cross-correlation - imshow -----
        # plot_2panels(cross_corr_arr, time_slice_arr, phi_lags, ylim=(-1500,1500),
        #              cbar_label='Amplitude', suptitle='Cross-correlation',
        #              save_name=f'/crosscorr_param_{p}', save_loc=SAVE_FIG_DIR)
    
        # ----- Cross-correlation - slices plot -----
        # plot_cross_corr_slices(phi_lags, cross_corr_arr, time,
        #                        true_RMs=rm_true,
        #                        save_name=f'/crosscorr_slices_param_{p}',
        #                        save_loc=SAVE_FIG_DIR
        #                        )

        # ----- Plot visible + detected bursts in FDF -----
        plot_bursts_in_fdf(p, stokes_arr[0], time_arr, phi_array, visible_dict, downsamp_list,
                            results_params_all_downsamp, param_label=PARAM_LABEL, rm_true=rm_true,
                            save_name=f'param_{p}', save_loc=BURST_VIS_DIR)


    # Time Curves
    # --------------------------------------------------------------------------
    # wall & cpu times vs param, grouped by block
    plot_timings_by_block(param_list, metadata, timings, param_label=PARAM_LABEL,
                        fig_title=None, plot_type='cpu_elapsed', save_name=f'/timecurves', save_loc=SAVE_FIG_DIR,
                        plot_total=True, plot_wall=False, plot_ylog=True)
    
    # cpu efficiency vs param, grouped by code block 
    plot_timings_by_block(param_list, metadata, timings, param_label=PARAM_LABEL,
                        fig_title=None, plot_type='cpu_efficiency', save_name=f'/efficiency', save_loc=SAVE_FIG_DIR,
                        plot_total=True, plot_ylog=False)
    

    # Plot total % detections (across all files) vs downsampling factor
    # --------------------------------------------------------------------------
    num_bursts_expected_tot = np.array([ 
        results_summary_all_downsamp[i]["num_bursts_expected_tot"] 
        for i in range(len(downsamp_list))
    ])
    num_pulses_visible_tot = np.array([ 
        results_summary_all_downsamp[i]["num_pulses_visible_tot"] 
        for i in range(len(downsamp_list))
    ])
    num_pulses_not_visible_tot = np.array([ 
        results_summary_all_downsamp[i]["num_pulses_not_visible_tot"] 
        for i in range(len(downsamp_list))
    ])
    num_pulses_detected_tot = np.array([ 
        results_summary_all_downsamp[i]["num_pulses_detected_tot"] 
        for i in range(len(downsamp_list))
    ])
    num_pulses_detected_notvisible_tot = np.array([ 
        results_summary_all_downsamp[i]["num_pulses_detected_notvisible_tot"] 
        for i in range(len(downsamp_list))
    ])

    plt.figure(figsize=(7,4))
    plt.scatter(downsamp_list, num_pulses_detected_tot/num_bursts_expected_tot *100, 
                label=r"$\frac{\text{detected}}{\text{expected}}$")
    plt.scatter(downsamp_list, num_pulses_detected_tot/num_pulses_visible_tot *100, 
                label=r"$\frac{\text{detected}}{\text{visible}}$")
    plt.scatter(downsamp_list, num_pulses_detected_notvisible_tot/num_pulses_not_visible_tot *100, 
                label=r"$\frac{\text{detected & not vis}}{\text{not vis}}$")
    plt.axhline(y=100, ls=':', color='grey', lw=1)

    plt.ylabel("% Detected")
    plt.xlabel("Downsampling Factor")
    plt.legend(title=f"time_tol = {TOL:.0f}", title_fontsize=9, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIG_DIR}/detection_rate_tol{TOL}.png", dpi=200)