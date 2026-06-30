# Scripts Documentation

This document describes the implementation of each Python script in the RM search pipeline, including processing steps, key functions used, inputs, outputs, and where results are saved.


## 01_rm_search.py

This python script is executed via `run_rm_search.sh`. This script is meant to be run as an array job, where ${STEP_LIST[$INDEX]} specifies the value of the parameter being varied. This could be the downsampling factor, the DM, the delay, the index of the first file to read in, etc. The start_file_index is what allows you to run the search on all the data for an observation night in chunks of nfiles. If the output directory is kept constant across array jobs, then the results of the search for a varied parameter will all be saved in the same directory, making it easy for scripts 02 and 03 to grab the data and analyze it all at once (or again as an array job). You can look at the output path section of `run_rm_search.sh` or the output directory examples to see how I kept track of different runs and how I organized my directories to work across all 3 scripts. 


### Inputs
- `source_dir`: Absolute path to the directory of the astrophysical source to analyze. This directory must follow the structure of CHIME raw baseband data (in VDIF format). 

  Expected structure:
  ```text
  {source_dir}/{freq_band}/{obs_night}_CHIME_vdif/{file_number}_0{freq_band}.vdif
- `obs_night`: Observation night to analyze for the chosen source.
- `save_file`: Absolute path and basename of file where to save the RM search results.


### Parameters
Descriptions of all parameters are documented in the script (with their default values) and can also be viewed by running `--help`. The following are all optional.

#### Source properties
- true_dm_pc_cm3
- true_rms_rad_m2
- delay_ns

#### Data parameters
- start_file_ind
- nfiles
- ref_freq
- npixels_to_avg
- slice_burst_flag

#### RM search configuration
- phi_max
- dphi_scaling
- rfi_mean_threshold
- rfi_std_threshold
- search_downsamp_factor
- compute_cross_corr_flag

#### Simulation parameters
These are used to add bursts on top of real data when the flag is set to True.

- sim_flag
- sim_num_bursts
- sim_arrival_times
- sim_burst_widths
- sim_snr
- sim_rm


**Note:** `npixels_to_avg` tells the code how much to downsample when going from baseband to Stokes data. Downsampling was initially done on the complex voltage signals, but with the current implementation, this downsampling happens later directly on the Stokes data. This means that the parameter is redundant with `search_downsamp_factor`, which tells the code how much to downsample the Stokes data before computing the FDF. Therefore, I suggest keeping `search_downsamp_factor=1` and using `npixels_to_avg` to control the downsampling since this reduces the data at an earlier step (and keeps the memory usage more reasonable). From the raw data, 391 times samples is what gives a resolution of ~1ms so it could also be simpler to keep npixels_to_avg set to 391 and use search_downsamp_factor in units of ms.


### Outputs
All outputs are saved in the same directory as `save_file`, with the following naming convention:

- `{save_file}_summary.txt`: High-level summary of RM search results.
- `{save_file}_arrays.npz`: Main data products (Stokes array, FDF, time axes, freqs, phi, etc)
- `{save_file}_metadata.npz`: Metadata associated with the run (parameters used, search configuration, data information)
- `{save_file}_timings.npz`: Timing and performance information for each step of the search.

The timings files have a nested structure that could be confusing initially. I suggest opening the results of one file and expanding + looking at the contents (dictionary elements, keys, items) to get a sense of how results are organized if it's not clear enough from the script itself. The function `plot_timings_by_block` in plot_utils.py also has a summary of the structure in its docstring.


### Functions from `rm_search_utils.py`
- print_mem(msg)
- read_stokes(...)
- flag_rfi_manual(ranges, freqs)
- get_rfi_mask(...)
- normalize_data(data_array)
- slice_around_burst(full_stokes, time, slice_width)
- invert_delay(delay, freq, U, V)
- get_spectra(Q, U, freq, ...)
- rm_synthesis(P_spec, phi_array, b, K)

See documentation for more information


### Functions from `sim_burst_utils.py`
- compute_timeseries_sigma(full_stokes)
- gen_sim_burst_params(...)
- gen_stokes_I(...)
- gen_stokes_QUV(...)
- apply_rm(I, Q, U, RM, freq)
- get_rot_stokes(...)

See documentation for more information



---
---

## 02_pulse_detection.py

This script processes all results contained within a single output directory produced by 01_rm_search.py and attempts to detect pulses in every FDF. This means that if the RM search was previously run as an array job over some varied parameter and the results are saved in a single output directory, this script will access all results over that varied parameter. Information about all detected pulses in each FDF is saved, as well as overall results across all runs. Therefore, the total counts only really make sense when the varied parameter in the search is the start_file_index.

This script can also be run as an array job, mainly to vary the downsampling factor but it should also work to vary the tolerance or prominence factor (see parameters section below) with only minor modifications. The main limitation is that the output subdirectories are currently hardcoded to use the `pulse_search_downsamp_factor` value to distinguish between array job results (see lines 79-101).


### Inputs
- `results_dir`: Absolute path to directory containing RM search results.
- `save_fig_dir`: Absolute path to directory where pulse detection results should be saved.
- `source_P0`: Pulse period (P0) of the source.
- `source_W50`: Pulse width (W50) of the source.
- `tol`: Minimum time separation between detected peaks in FDF to be classified as separate pulses. 


### Parameters
- pulse_search_downsamp_factor: controls how much **the FDF** is averaged before stepping through each time sample to find peaks along the phi axis. _Do not confuse with npixels_to_avg or search_downsamp_factor from the search script_
- prominence_factor: parameter used in scipy.find_peaks() to find peaks in FDF.
- plot_detected_pulses: flag to produce all pulse plots (see `start_file_ind_dm141/tol200/burst_detected_downsamp4_tol200`) in the example figures directory.


### Outputs 
This script produces a number of result files (.npz and .txt) as well as plots of the pulses found. All outputs are saved either directly in the given `save_fig_dir` or in subdirectories automatically created by the script. The naming convention used is:

- `{save_fig_dir}/burst_visible_info.npz`: contains information about pulses visible in Stokes I.
- `{save_fig_dir}/tol{tol_value}/burst_detected_downsamp{pulse_search_downsamp_factor}_tol{tol_value}`: subdirectory for burst detection plots
- `{save_fig_dir}/tol{tol_value}/burst_detected_info_downsamp{pulse_search_downsamp_factor}_tol{tol_value}.npz`: file to save detected bursts info
- `{save_fig_dir}/tol{tol_value}/summary_downsamp{pulse_search_downsamp_factor}_tol{tol_value}.txt`: summary file

See comment at end of 02_pulse_detection.py for more information on .npz output format.

**Note:** When this script is run as an array job, all runs use the same Stokes data, while only the pulse detection parameters (e.g., pulse_search_downsamp_factor, tol, or prominence_factor) change. As a result, the visible pulses in Stokes I are identical for every job and only need to be saved once. To avoid multiple jobs attempting to write the same file simultaneously, burst_visible_info.npz is only written when `PULSE_SEARCH_DOWNSAMP_FACTOR == 4` (currently hardcoded; see line 335). The visible pulses are still computed by all simultaneous jobs in the first set of the array job, but only one job saves the results. Jobs check whether the file already exists, and if it does they simply load the data, meaning that all subsequent jobs do not recompute the visible pulses.


### Functions from `pulse_utils.py`
- count_visible_pulses(...)
- downsample_time_mean(data, time, factor)
- separate_pulses(detections, time_tol)
- plot_pulses(...)
- find_pulses_fdf(...)
- count_found_notvisible(detected_bursts, visible_groups, time_arr_vis)



---
---

## 03_plot_search_results.py

This script uses the results produced by the two previous scripts to generate a variety of plots, including Stokes data, FDFs, detected pulses, timing distributions, and other summary figures. The script assumes the directory structure and output file organization created by the previous scripts, and expects the required .npz files to be present for every result (it will fail if any are missing).

This is meant to be used to plot the results for a single output directory from the RM search (e.g., when the RM search is run as an array job over the start file index) and compare multiple pulse detection results (e.g., when varying the pulse search downsampling factor, tolerance, or other pulse detection parameters). 


### Inputs
- `results_dir`: Absolute path to directory containing RM search results.
- `save_fig_dir`: Absolute path to directory containing pulse detection results and where figures will be saved.
- `tol`: Tolerance value to plot results for. This is used to access results using the given naming convention.

Both results_dir and save_fig_dir must match the directory structure of scripts 01 and 02. Some assumptions about the directory structure are built into the script and are all located before line 122 in the code. If/when using a different directory structure (or when varying something other than pulse_search_downsamp_factor in script 02), make sure the code reflects those changes.


### Parameters
- param_label: Label for the x-axis corresponding to the varied parameter in the RM search.
- data_files_unique: flag to tell the code if the Stokes data is the same across all RM search runs or not.
- plot_cross_corr: flag to produce or skip the cross-correlation plots.


### Outputs
Figures are saved in various subdirectories created within `save_fig_dir`. All the following output paths are relative to `save_fig_dir` and unless specified otherwise, one plot is made per value of the varied parameter in the RM search.

- Stokes data plots: Saved in `stokes/`. Created once only when data_files_unique is set to True.
- FDF plot(s): Saved in `FDFs/`. Only produced when the subdirectory does not exist yet.
- RMSF plot(s): Saved in `RMSFs/`. Created once only when data_files_unique is set to True.
- Cross-correlation plots: Saved in `crosscorr/`. Not produced when plot_cross_corr is set to False.
- Stokes I S/N with phi detections in FDF: Saved in `tol{tol_value}/burst_visible_stokesI_tol{tol_value}/`. Each plot contains results from all the corresponding runs of 02_pulse_detection.py (eg. all downsampling factors) for a given tol value. 
- Timings plots: saved directly in save_fig_dir. Plots the timing results for all steps in the RM search as a function of the varied parameter.
- Detection rate plot: Saved directly in save_fig_dir. This plot combines information from all RM search runs and plots the results as a function of the varied parameter in the pulse detection script. 


### Functions from `plot_utils.py`
- plot_stokes(...)
- plot_rmsf(...)
- plot_2panels(...)
- plot_cross_corr_slices(...)
- scale_lightness(rgb, scale_l)
- get_colors(num_colors)
- plot_timings_by_block(...)
- plot_bursts_in_fdf(...)


