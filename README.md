# RM Search Pipeline 

## Overview

This repo contains code for performing a polarimetric search on CHIME baseband data using RM synthesis.

The project is organized into three main steps:

1. RM search
2. Pulse detection
3. Plots



## Running the code

Disclaimer: This code has been implemented to run on the trillium cluster and some changes might be needed if running elsewhere.

Each step has a python script which is accompanied by a corresponding shell script for launching (array) jobs. The `.sh` scripts handle parameter configuration, input/output paths, and SLURM job setup (including array jobs), so that the user can run each step by simply submitting `sbatch <script_name>.sh`. See `scripts/README.md` for a detailed description of each script, inputs, outputs, and saved products.

See `requirements.txt` for necessary packages.


### RM Search

This step performs the RM search and produces the Faraday Dispersion Function (FDF).

Bash script: `run_rm_search.sh`

Python script: `scripts/01_rm_search.py`

Functions: `scripts/rm_search_utils.py` and `scripts/sim_burst_utils.py`


### Pulse Detection

This step identifies and groups significant FDF detections into distinct pulses.

Bash script: `run_pulse_detection.sh`

Python script: `scripts/02_pulse_detection.py`

Functions: `scripts/pulse_utils.py`


### Plots

Generates figures and summary plots from the search results and pulse detection.

Bash script: `run_plots.sh`

Python script: `scripts/03_plot_search_results.py`

Functions: `scripts/plot_utils.py`



### Examples
`output_directory_examples` contains an example of the directory structure I used and where different files were saved. Some folders are empty and except a few .txt files, no file actually contains data. Some example figures are given in `timeavg391_nfiles300_allfiles_RFI1.7mean1.3std/SUBDIR_dm141`.