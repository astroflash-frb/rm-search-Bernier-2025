#!/bin/bash
#SBATCH --job-name=search_plots
#SBATCH --account=def-istairs
#SBATCH --time=01:00:00
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-0%1
#SBATCH --output=/scratch/abernier/logs/scripts/%j_%a.out
#SBATCH --mail-user=audreanne.bernier@mail.mcgill.ca
#SBATCH --mail-type=ALL

# Load modules
cd ~/scratch
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2024b
echo "Modules loaded"

# Load python environment
source /home/abernier/envs/tril_env/bin/activate
echo "Environment loaded"


# Source info
SOURCE=2022_CHIME_B2111+46
OBS_NIGHT=20220826T064420Z

# Input & output paths (assuming same hierarchy as search script output)
SETTINGS=timeavg391_nfiles300_allfiles_RFI1.7mean1.3std
SUBDIR=start_file_ind_dm135
RESULTS_DIR=/scratch/abernier/results_data/${SOURCE}/${OBS_NIGHT}/${SETTINGS}/${SUBDIR}  # input
SAVE_FIG_DIR=/scratch/abernier/results_figures/${SOURCE}/${OBS_NIGHT}/${SETTINGS}/${SUBDIR}   # input+output
mkdir -p "$SAVE_FIG_DIR"  # create output directory if it doesn't exist

# Info for plots
PARAM_LABEL="Start File Index"  # Label for the x-axis corresponding to the varied parameter (e.g., 'Downsampling Factor', 'DM [pc/cm^3]')
DATA_FILES_UNIQUE=0  # 1 to plot stokes only once, 0 to plot stokes for each param value 
TOL=200  # tol to plot results for


# Run plotting script
cd /home/abernier/scratch/rm-search-Bernier-2025/scripts
echo "03_plot_search_results.py ${RESULTS_DIR} ${SAVE_FIG_DIR} ${TOL}\
      --param_label \"${PARAM_LABEL}\" \
      --data_files_unique ${DATA_FILES_UNIQUE}"
python 03_plot_search_results.py ${RESULTS_DIR} ${SAVE_FIG_DIR} ${TOL}\
      --param_label "${PARAM_LABEL}" \
      --data_files_unique ${DATA_FILES_UNIQUE}