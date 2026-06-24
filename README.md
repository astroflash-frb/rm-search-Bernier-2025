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


### RM Search

This step performs the RM search and produces the Faraday Dispersion Function (FDF).

Bash script: `run_rm_search.sh`

Python script: `01_rm_search.py`

Functions: `rm_search_utils.py` and `sim_burst_utils.py`


### Pulse Detection

This step identifies and groups significant FDF detections into distinct pulses.

Bash script: `run_pulse_detection.sh`

Python script: `02_pulse_detection.py`

Functions: `pulse_utils.py`


### Plots

Generates figures and summary plots from the search results.

Bash script: `run_plots.sh`

Python script: `03_plot_search_results.py`

Functions: `plot_utils.py`



