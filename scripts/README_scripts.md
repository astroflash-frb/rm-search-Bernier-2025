# Scripts Documentation

This document describes the implementation of each Python script in the RM search pipeline, including processing steps, key functions used, inputs, outputs, and where results are saved.


## 01_rm_search.py

This python script is executed via `run_rm_search.sh`. This script is meant to be run as an array job over some varied parameter (using `${STEP_LIST[$INDEX]}`). This could be the downsampling factor, the DM, the delay, the index of the first file to read in, etc. The start_file_index is what allows you to run the search on all the data for an observation night in chunks of nfiles. If the outdir is kept constant across array jobs, then the results of the search for a varied parameter will all be saved in the same directory, making it easy for scripts 02 and 03 to grab the data and analyze it all at once (or again as an array job). You can look at the output path section of `run_rm_search.sh` or the output directory examples to see how I kept track of different runs and how I organized my directories to work across all 3 scripts. 


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
These are used to add bursts on top of real data.

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

This script processes all results contained within a single output directory produced by 01_rm_search.py and attempts to detect pulses in every FDF. This means that if the RM search is run as an array job over some varied parameter and the results are saved in the same outdir, this script will access all results over that varied parameter. Information about all detected pulse in each FDF are saved, as well as overall results across all runs. Therefore, the total counts only really make sense when the varied parameter in the search is the start_file_index.

This script can also be run as an array job, mainly to vary the downsampling factor but it should also work to vary the tolerance or prominence factor (see parameters section below) with only minor modifications. The main thing is that the output subdirectories for all the different results of the array job are hardcoded to use the `pulse_search_downsamp_factor` value to differentiate them.


### Inputs
- `results_dir`: Absolute path to directory containing RM search results.
- `save_fig_dir`: Absolute path to directory where pulse detection results should be saved.
- `source_P0`: Pulse period (P0) of the source.
- `source_W50`: Pulse width (W50) of the source.
- `tol`: Minimum time separation between detected peaks in FDF to be classified as separate pulses. 


### Parameters
- pulse_search_downsamp_factor: controls how much the FDF is averaged before stepping through each time sample to find peaks along the phi axis.
- prominence_factor: parameter used in scipy.find_peaks() to 
- plot_detected_pulses: flag to produce all pulse plots (see `start_file_ind_dm141/tol200/burst_detected_downsamp4_tol200`) in the example figures directory.


### Outputs 
All outputs are saved either directly in the input `save_fig_dir` or in subdirectories automatically created by the script. The naming convention is:

- `{save_fig_dir}/burst_visible_info.npz`: contains information about pulses visible in Stokes I.
- `{save_fig_dir}/tol{tol_value}/burst_detected_downsamp{pulse_search_downsamp_factor}_tol{tol_value}`: subdirectory for burst detection plots
- `{save_fig_dir}/tol{tol_value}/burst_detected_info_downsamp{pulse_search_downsamp_factor}_tol{tol_value}.npz`: file to save detected bursts info
- `{save_fig_dir}/tol{tol_value}/summary_downsamp{pulse_search_downsamp_factor}_tol{tol_value}.txt`: summary file


See comment at end of 02_pulse_detection.py for more information on output format.

**Note:** When this script is run as an array job, all runs use the same Stokes data, while only the pulse detection parameters (e.g., pulse_search_downsamp_factor, tol, or prominence_factor) change. As a result, the visible pulses in Stokes I are identical for every job and only need to be saved once. To avoid multiple jobs attempting to write the same file simultaneously, burst_visible_info.npz is only written when PULSE_SEARCH_DOWNSAMP_FACTOR == 4 (currently hardcoded; see line 335). The visible pulses are still computed by all simultaneous jobs in the first set of the array job, but only one job saves the results. Jobs check whether the file alredy exists, and if it does they simply load the data, meaning that all subsequent jobs do not recompute the visible pulses.


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

### Inputs


### Parameters


### Outputs


### Functions from `plot_utils.py`

