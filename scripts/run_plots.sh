#!/bin/bash
#SBATCH --job-name=search_plots
#SBATCH --account=def-istairs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-3%3
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

INDEX=$((SLURM_ARRAY_TASK_ID - 1))
LIST=(4 8 16)

# Source info
SOURCE=2022_CHIME_B2111+46
OBS_NIGHT=20220826T064420Z
SOURCE_P0=1.014684793189  # pulse period in seconds
SOURCE_W50=60.5  # pulse width in ms
PULSE_SEARCH_DOWNSAMP_FACTOR=${LIST[$INDEX]}  # downsampling factor to apply to FDF_arr before pulse detection
PROMINENCE_FACTOR=10  # prominence cutoof factor to apply when finding peaks in FDF
TOL=100  # min distance bw peaks for them to be counted as 2 peaks (in ms)

# RM search results info
RESULTS_DIR=/scratch/abernier/results_data/${SOURCE}/${OBS_NIGHT}
SETTINGS=timeavg391_nfiles300_allfiles_RFI1.7mean1.3std
PARAM=start_file_ind  # Specify which parameter was varied in the search results to plot
PARAM_LABEL="Start File Index"  # Label for the x-axis corresponding to the varied parameter (e.g., 'Downsampling Factor', 'DM [pc/cm^3]')
BURST_LIMS="0"  # burst limits in time
DATA_FILES_UNIQUE=0  # 1 to plot stokes only once, 0 to plot stokes for each param value 

# plot stokes and fdf only for the first job since using save search results
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    PLOT_FDF=1
else
    PLOT_FDF=0
fi

# Run plotting script
cd /home/abernier/scratch/rm-search-Bernier-2025/scripts
echo "03_plot_search_results.py ${RESULTS_DIR} ${SETTINGS} ${PARAM} ${SOURCE_P0} ${SOURCE_W50} ${TOL}\
      --pulse_search_downsamp_factor ${PULSE_SEARCH_DOWNSAMP_FACTOR} \
      --prominence_factor ${PROMINENCE_FACTOR} \
      --param_label \"${PARAM_LABEL}\" \
      --burst_lims ${BURST_LIMS} \
      --data_files_unique ${DATA_FILES_UNIQUE} \
      --plot_fdf ${PLOT_FDF}"
python 03_plot_search_results.py ${RESULTS_DIR} ${SETTINGS} ${PARAM} ${SOURCE_P0} ${SOURCE_W50} ${TOL}\
      --pulse_search_downsamp_factor ${PULSE_SEARCH_DOWNSAMP_FACTOR} \
      --prominence_factor ${PROMINENCE_FACTOR} \
      --param_label "${PARAM_LABEL}" \
      --burst_lims ${BURST_LIMS} \
      --data_files_unique ${DATA_FILES_UNIQUE} \
      --plot_fdf ${PLOT_FDF}