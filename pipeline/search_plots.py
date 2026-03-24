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
parser.add_argument("settings", type=str, 
                    help='Specify which Stokes data was used in the search.')
parser.add_argument("param", type=str, 
                    help='Sepecify parameter that was varied in the search results to plot.')
parser.add_argument(
    "-r", 
    "--rm_true", 
    type=float, 
    nargs='+', 
    default=0.,  
    help='True RM value(s) to plot as vertical line(s) on the ross-correlation slices plot.'
)
args = parser.parse_args()

# Info to get search results
SETTINGS = args.settings
PARAM = args.param
DIR = f"/home/abernier/scratch/pulsar_data/search_results/{SETTINGS}/{PARAM}/"

# Search result files
METADATA_FILES = glob.glob(os.path.join(DIR, f"2022_CHIME_B2111+46_20220826T064420Z_{PARAM}*_metadata.npz"))
DATA_FILES = glob.glob(os.path.join(DIR, f"2022_CHIME_B2111+46_20220826T064420Z_{PARAM}*_arrays.npz"))
NFILES = len(DATA_FILES)
DATA_FILES_UNIQUE = True   # determines if masked+normalized stokes data is the same for all values of the varied params or not
                           # True if data is unique, false if data changes with each run of param
RM_TRUE = args.rm_true

# Save directory for figures
SAVE_FIG_DIR = f'/scratch/abernier/rm-search-Bernier-2025/figures/{SETTINGS}/{PARAM}/'
os.makedirs(os.path.dirname(SAVE_FIG_DIR), exist_ok=True)



# Load Functions
# ------------------------------------------------------------------------------
def load_data(file, param):
    param_value = int(file.split(f"{param}")[2].split("_")[0])  # Extract param value from filename
    with np.load(file, allow_pickle=True) as f:
        data_dict = {k: f[k] for k in f.files}
    return param_value, data_dict

def get_data(files, param):
    # ---- Load and sort files ----
    results = [load_data(f, param) for f in files]
    param_list, dict_list = zip(*results) 

    sort_index = np.argsort(param_list)
    param_list = np.array(param_list)[sort_index]  # list of values ran for the chosen parameter 
    dict_list = [dict_list[i] for i in sort_index]  # list of data dict for each param

    # ---- Initialize lists ----
    stokes_list = []
    time_list = []
    freq_list = []
    RMSF_list = []
    RMSF_full_list = []
    lambda2_list = []
    rfi_mask_list = []

    phi_list = []
    FDF_list = []
    RM_list = []
    time_slice_list = []
    cross_corr_list = []
    phi_lags_list = []

    # ---- Fill lists ----
    for i, data_dict in enumerate(dict_list):
        # only read in these once if they are the same between parameter runs
        if not DATA_FILES_UNIQUE or i==0:
            stokes_list.append(data_dict['stokes_norm_masked'])
            time_list.append(data_dict['time'])
            freq_list.append(data_dict['freq'])
            RMSF_list.append(data_dict['RMSF'])
            RMSF_full_list.append(data_dict['RMSF_full'])
            lambda2_list.append(data_dict['lambda2_array'])
            rfi_mask_list.append(data_dict['rfi_mask'])

        phi_list.append(data_dict['phi_array'])
        FDF_list.append(data_dict['FDF_arr'])
        RM_list.append(data_dict['RM_meas_arr'])
        time_slice_list.append(data_dict['time_slice_arr'])
        cross_corr_list.append(data_dict['cross_corr_arr'])
        phi_lags_list.append(data_dict['phi_lags'])

    # ---- Return everything as lists ----
    return {
        "param_list": param_list,
        "stokes": stokes_list,
        "time": time_list,
        "freq": freq_list,
        "RMSF": RMSF_list,
        "RMSF_full": RMSF_full_list,
        "lambda2": lambda2_list,
        "rfi_mask": rfi_mask_list,
        "phi": phi_list,
        "FDF": FDF_list,
        "RM": RM_list,
        "time_slice": time_slice_list,
        "cross_corr": cross_corr_list,
        "phi_lags": phi_lags_list,
    }


def load_metrics(file, param):
    param_value = int(file.split(f"{param}")[2].split("_")[0])  # Extract param value from filename
    metadata_dict = np.load(file, allow_pickle=True)['metadata'].item()
    
    return param_value, metadata_dict

def get_metric_dict_list(files, param):
    # open files and get results
    results = [load_metrics(f, param) for f in files]
    param_list, dict_list = zip(*results)
    
    # sort
    sort_index = np.argsort(param_list)
    param_list = np.array(param_list)[sort_index]  # list of values ran for the chosen parameter 
    dict_list = np.array(dict_list)[sort_index]  # list of dicts for each param

    return param_list, dict_list



if __name__ == "__main__":
    # Open Data & Metadata
    # ------------------------------------------------------------------------------
    # Open data from all files
    data = get_data(DATA_FILES, PARAM)
    param_list = data["param_list"]

    # THESE ARE ALL LISTS OF ARRAYS, shape is given by each element
    # Stokes data (lists of arrays, one per file)
    stokes_norm_masked = data['stokes']  # shape (Nstokes, Ntimes, Nfreqs)
    time = data['time']  # shape (Ntimes,)
    freq = data['freq']  # shape (Nfreqs,)
    rfi_mask = data['rfi_mask']  # shape (Nfreqs,)

    # RMSF
    RMSF = data['RMSF']  # shape (Nphi,)
    RMSF_full = data['RMSF_full']  # shape (Nphi,)
    lambda2_array = data['lambda2']  # shape (Nlambda,) = (Nfreq,)

    # Search results (lists of arrays)
    phi_array = data['phi']  # shape (Nphi,)
    FDF_arr = data['FDF']  # shape (Ntimes_fdf, Nphi)
    RM_meas_arr = data['RM']  # shape (Ntimes_fdf,)
    time_slice_arr = data['time_slice']  # gives time avg array for FDFs, shape (Ntimes_fdf,) ** might not all be equal
    cross_corr_arr = data['cross_corr']  # shape (Ntimes_fdf, Nphi)
    phi_lags = data['phi_lags']  # shape (Nlags,)

    # Get Metadata and time data
    _, dict_list = get_metric_dict_list(METADATA_FILES, PARAM)
    # Hierarchy of dict_list:
    # - every element in the list is a dict for different run ie param value
    # - within those dicts, there are different dicts for each block of code. 
    #    These have time information 'total', 'load_mask_normalize', 'rmsf_computation', 'rm_search', 'cross_correlation'

    # Printing info
    print(dict_list[0].keys())
    print(dict_list[0]['timings'].keys())
    print(dict_list[0]['timings']['total'])


    # Plots
    # ------------------------------------------------------------------------------
    # Plot Data
    for i in range(NFILES):
        # Get labels
        if DATA_FILES_UNIQUE:
            label_stokes = 'stokes_data'
            label_rmsf = 'RMSF'
        else:
            label_stokes = f'stokes_data_{PARAM}{param_list[i]}'
            label_rmsf = f'RMSF_{PARAM}{param_list[i]}'
            
        # Plot masked + normalized Stokes data
        plot_stokes(stokes_norm_masked[i], freq=freq[i], time=time[i], cbar_lim_max=0.5,
                    t_unit='s', save_name=label_stokes, save_loc=SAVE_FIG_DIR)

        # Plot RMSF
        plot_rmsf(phi_array[i], RMSF[i], RMSF_full[i], save_name=label_rmsf, save_loc=SAVE_FIG_DIR)

        if DATA_FILES_UNIQUE:  # if data is the same for all runs, only plot once
            break

    
    # Plot FDF phi/time image for each search parameter
    for i in range(NFILES):
        plot_2panels(FDF_arr[i], time_slice_arr[i], phi_array[i], 
                     cbar_label='Amplitude', suptitle='FDF', 
                     save_name=f'FDF_{PARAM}{param_list[i]}', save_loc=SAVE_FIG_DIR)
        
    
    # Plot cross-correlations for each parameter
    for i in range(NFILES):
        plot_2panels(cross_corr_arr[i], time_slice_arr[i], phi_lags[i], ax2_ylabel=r'$\phi$ [rad/m$^2$]', 
                     cbar_label='Amplitude', ax1_type='peak', suptitle='Cross-correlation', ylim=(-1500,1500),
                     save_name=f'crosscorr_{PARAM}{param_list[i]}', save_loc=SAVE_FIG_DIR)
    

    # Plot cross-correlation slices
    for i in range(NFILES):
        plot_cross_corr_slices(phi_lags[i], cross_corr_arr[i], 
                               true_RM = RM_TRUE,
                               save_name=f'crosscorr_slices_{PARAM}{param_list[i]}',
                               save_loc=SAVE_FIG_DIR
                               )
    

    # Time Curves
    # ------------------------------------------------------------------------------
    # wall & cpu times vs param, grouped by block
    plot_time_curves_by_block(param_list, dict_list, 'Downsampling Factor', 
                              fig_title=None, save_name='timecurves', save_loc=SAVE_FIG_DIR,
                              plot_total=True, plot_wall=True, plot_ylog=True, convert_tstep=True)
    
    # cpu efficiency vs param, grouped by code block 
    plot_efficiency_by_block(param_list, dict_list, 'Downsampling Factor', 
                             fig_title=None, save_name='efficiency', save_loc=SAVE_FIG_DIR,
                             plot_total=True, plot_ylog=False, convert_tstep=True)