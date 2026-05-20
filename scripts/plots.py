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
parser.add_argument("--param_label", type=str, default=None,
                    help='Label for the x-axis corresponding to the varied parameter (e.g., "Downsampling Factor", "DM [pc/cm^3]").')
parser.add_argument("--burst_lims", nargs='+', type=float, default=None,
                    help='Time limits (in seconds) to average over burst in some plots. Provide as two numbers: --burst_lims tmin tmax')
parser.add_argument("--data_files_unique", action='store_true',
                    help='Flag to indicate whether the masked+normalized Stokes data is ' \
                    'the same for all values of the varied parameter (True) or if it changes ' \
                    'with each run (False).')
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


# Print info to summary file (overwrite)
# ------------------------------------------------------------------------------
summary_file = f"{RESULTS_DIR}/{SETTINGS}/{PARAM}_plot_summary.txt"
with open(summary_file, "w") as SUMMARY_FILE:

    print("=" * 80, file=SUMMARY_FILE)
    print("RM SEARCH PLOT SUMMARY", file=SUMMARY_FILE)
    print("=" * 80, file=SUMMARY_FILE)

    # General info
    print("\n[RUN INFO]", file=SUMMARY_FILE)
    print(f"Results directory : {RESULTS_DIR}", file=SUMMARY_FILE)
    print(f"Settings          : {SETTINGS}", file=SUMMARY_FILE)
    print(f"Varied parameter  : {PARAM}", file=SUMMARY_FILE)
    print(f"Number of runs    : {NFILES}", file=SUMMARY_FILE)



if __name__ == "__main__":
    # Open Results
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

    
    # Print info to summary file
    # ------------------------------------------------------------------------------
    with open(summary_file, "a") as SUMMARY_FILE:
        # Parameter info
        print(f"Parameter values  : {sorted(param_list)}", file=SUMMARY_FILE)

        # Metadata contents
        first_param = sorted(param_list)[0]
        print("\n[METADATA CONTENTS]", file=SUMMARY_FILE)
        print("Stored per-run search settings and results.", file=SUMMARY_FILE)
        for k in metadata[first_param].keys():
            print(f"  - {k}", file=SUMMARY_FILE)

        # Timing contents
        print("\n[TIMING CONTENTS]", file=SUMMARY_FILE)
        print("Stored runtime diagnostics for each processing stage.", file=SUMMARY_FILE)
        for k in timings[first_param].keys():
            print(f"  - {k}", file=SUMMARY_FILE)

        # Array contents
        print("\n[DATA ARRAYS]", file=SUMMARY_FILE)
        print(f"Example of arrays available for plotting (using {PARAM}={first_param}):", file=SUMMARY_FILE)
        for k, arr in data[first_param].items():
            try:
                print(
                    f"  - {k:<25} "
                    f"shape={str(arr.shape):<18} "
                    f"dtype={arr.dtype}",
                    file=SUMMARY_FILE
                )
            except Exception:
                print(f"  - {k}", file=SUMMARY_FILE)
        
        print("\n[PLOTS]", file=SUMMARY_FILE)


    # Plot Data
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
        RMSF = data[p]['RMSF']  # shape (Nphi,)
        RMSF_full = data[p]['RMSF_full']  # shape (Nphi,)
        phi_array = data[p]['phi_array']  # shape (Nphi,)
        plot_rmsf(phi_array, RMSF, RMSF_full, save_name=label_rmsf, save_loc=SAVE_FIG_DIR)

        if DATA_FILES_UNIQUE:  # if data is the same for all runs, only plot once
            break

    
    # -----------------------------------------------------------------------------
    # Get true RMs (using 1st file)
    rm_true = list(metadata[param_list[0]]['true_rms_rad_m2'].value)  # true RM of real data
    if metadata[param_list[0]]['sim_flag']:
        sim_rm = metadata[param_list[0]]['sim_params']["rm"]  # true RM of injected bursts
        rm_true.append(sim_rm.value)
    rm_true = np.array(rm_true)  # shape (Ntrue_rms,)

    
    # Plot FDF and cross-correlation results
    # ------------------------------------------------------------------------------
    for p in param_list:
        # Get arrays to plot
        FDF_arr = data[p]['FDF_arr']  # shape (Ntime_fdf, Nphi)
        time_slice_arr = data[p]['time_slice_arr']  # shape (Ntime_fdf,)
        phi_array = data[p]['phi_array']  # shape (Nphi,)
        cross_corr_arr = data[p]['cross_corr_arr']  # shape (Ntime_fdf, Nphi_lags)
        phi_lags = data[p]['phi_lags']  # shape (Nphi_lags,)

        # FDF - imshow
        plot_2panels(FDF_arr, time_slice_arr, phi_array, 
                     cbar_label='Amplitude', suptitle='FDF', 
                     save_name=f'FDF_{PARAM}_{p}', save_loc=SAVE_FIG_DIR)
        
        # Cross-correlation - imshow
        plot_2panels(cross_corr_arr, time_slice_arr, phi_lags, ylim=(-1500,1500),
                     cbar_label='Amplitude', suptitle='Cross-correlation',
                     save_name=f'crosscorr_{PARAM}_{p}', save_loc=SAVE_FIG_DIR)
    
        # Cross-correlation - slices plot
        plot_cross_corr_slices(phi_lags, cross_corr_arr, 
                               true_RMs=rm_true,
                               save_name=f'crosscorr_slices_{PARAM}_{p}',
                               save_loc=SAVE_FIG_DIR
                               )
        

    # Folded FDF plot - MULTIPLE PANELS (for different values of varied param)
    # ------------------------------------------------------------------------------
    # with open(summary_file, 'a') as f:
    #         print(f"Plotting folded FDF panels for {len(param_list)} values of {PARAM}: {param_list}", file=f)

    # # Plot using burst time limits (if given)
    # if BURST_LIMS is not None:
    #     with open(summary_file, "a") as SUMMARY_FILE:
    #         print(f"Input time limits: {BURST_LIMS}", file=SUMMARY_FILE)
    #         print("Actual time limits for folded FDF:", file=SUMMARY_FILE)
        
    #     plot_folded_fdf(data, param_list, param_name=PARAM, summary_file=summary_file,
    #                     lim_time=BURST_LIMS, true_RMs=rm_true, xmax=500,
    #                     save_name=f"FDF_folded_allpanels_burstlims", save_loc=SAVE_FIG_DIR)
        
    # # Plot using full time range (no limits)
    # with open(summary_file, "a") as SUMMARY_FILE:
    #     print("Plotting folded FDF panels with no time limits (i.e. using full FDF time range for folding).", file=SUMMARY_FILE)
    
    # plot_folded_fdf(data, param_list, param_name=PARAM, summary_file=summary_file,
    #                 lim_time=None, true_RMs=rm_true, xmax=500,
    #                 save_name=f"FDF_folded_allpanels", save_loc=SAVE_FIG_DIR)



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
    

    # Get SNR arrays
    # ------------------------------------------------------------------------------

    # SNR for FDF
    # fdf_snr_arr = [
    #     compute_max_snr_per_burst(
    #         data = data[p]['FDF_arr'],  # shape (Ntime, Nphi), 
    #         phi_arr = data[p]['phi_array'],  # shape (Nphi,)
    #         rm_true = rm_true
    #     )
    #     for p in param_list
    # ]
    # fdf_snr_arr = np.array(fdf_snr_arr)   # shape [Nparams, Nbursts]

    # SNR for folded FDF (using burst time limits)
    # snr_folded_arr = None
    # if BURST_LIMS is not None:
    #     snr_folded_arr = [
    #         compute_max_snr_per_burst(
    #             data = data[p]['FDF_arr'],  # shape (Ntime, Nphi)
    #             phi_arr = data[p]['phi_array'],  # shape (Nphi,)
    #             rm_true = abs(rm_true),
    #             folded=True
    #         ) 
    #         for p in param_list
    #     ]
    #     snr_folded_arr = np.array(snr_folded_arr)   # shape [Nparams, Nbursts]



    # Plot CPU and S/N
    # plot_cpu_and_snr(param_list=param_list, metadata_dict=metadata, timings_dict=timings, 
    #                  snr_arr=fdf_snr_arr, folded_snr_arr=snr_folded_arr,
    #                  rm_true=rm_true,
    #                  param_label=PARAM_LABEL, 
    #                  save_name="cpu_and_snr",
    #                  save_loc=SAVE_FIG_DIR
    #                 )