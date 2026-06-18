#!/bin/bash
#SBATCH --job-name=pulse_detect
#SBATCH --account=def-istairs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-7%4
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
SOURCE_P0=1.014684793189  # pulse period in seconds
SOURCE_W50=60.5  # pulse width in ms

# Parameters for pulse detection in FDF
LIST=(2 4 6 8 10 12 14 16)
INDEX=$((SLURM_ARRAY_TASK_ID))  # index for array job to specify which parameter value to use for this job
PULSE_SEARCH_DOWNSAMP_FACTOR=${LIST[$INDEX]}  # downsampling factor to apply to FDF_arr before pulse detection
PROMINENCE_FACTOR=10  # prominence cutoof factor to apply when finding peaks in FDF
TOL=200  # min distance bw peaks for them to be counted as 2 peaks (in ms)
PLOT_DETECTED_PULSES=1  # whether to save plots of detected pulses in FDF for each parameter value (1 to save, 0 to not save)

# Input & output paths (assuming same hierarchy as search script output)
SETTINGS=timeavg391_nfiles300_allfiles_RFI1.7mean1.3std
SUBDIR=start_file_ind_dm141
RESULTS_DIR=/scratch/abernier/results_data/${SOURCE}/${OBS_NIGHT}/${SETTINGS}/${SUBDIR}  # input
SAVE_FIG_DIR=/scratch/abernier/results_figures/${SOURCE}/${OBS_NIGHT}/${SETTINGS}/${SUBDIR}   # output
mkdir -p "$SAVE_FIG_DIR"  # create output directory if it doesn't exist


# Run plotting script
cd /home/abernier/scratch/rm-search-Bernier-2025/scripts
echo "02_pulse_detection.py ${RESULTS_DIR} ${SAVE_FIG_DIR} ${SOURCE_P0} ${SOURCE_W50} ${TOL} \n\
      --pulse_search_downsamp_factor ${PULSE_SEARCH_DOWNSAMP_FACTOR} \n\
      --prominence_factor ${PROMINENCE_FACTOR} \n\
      --plot_detected_pulses ${PLOT_DETECTED_PULSES}"
python 02_pulse_detection.py ${RESULTS_DIR} ${SAVE_FIG_DIR} ${SOURCE_P0} ${SOURCE_W50} ${TOL} \
      --pulse_search_downsamp_factor ${PULSE_SEARCH_DOWNSAMP_FACTOR} \
      --prominence_factor ${PROMINENCE_FACTOR} \
      --plot_detected_pulses ${PLOT_DETECTED_PULSES}