# Scripts Documentation

This document describes the implementation of each Python script in the RM search pipeline, including processing steps, key functions used, inputs, outputs, and where results are saved.


## 01_rm_search.py

This python script is executed via `run_rm_search.sh`.

### Inputs
- `source_dir`: Absolute path to the directory of the astrophysical source to analyze. This directory must follow the structure of CHIME raw baseband data (in VDIF format). 

  Expected structure:
  ```text
  {source_dir}/{freq_band}/{obs_night}_CHIME_vdif/{file_number}_0{freq_band}.vdif
- `obs_night`: Observation night to analyze for the chosen source.
- `save_file`: Absolute path and basename of file where to save the RM search results.


### Parameters
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
- sim_flag
- sim_num_bursts
- sim_arrival_times
- sim_burst_widths
- sim_snr
- sim_rm


### Outputs
All outputs are saved in the same directory as `save_file`, with the following naming convention:

- `{save_file}_summary.txt`: High-level summary of RM search results.
- `{save_file}_arrays.npz`: Main data products (Stokes array, FDF, time axes, freqs, phi, etc)
- `{save_file}_metadata.npz`: Metadata associated with the run (parameters used, search configuration, data information)
- `{save_file}_timings.npz`: Timing and performance information for each step of the search.


### Functions from `rm_search_utils.py`

### Functions from `sim_burst_utils.py`





## 02_pulse_detection.py




## 03_plot_search_results.py

