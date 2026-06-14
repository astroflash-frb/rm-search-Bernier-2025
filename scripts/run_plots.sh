#!/bin/bash
#SBATCH --job-name=search_plots
#SBATCH --account=def-istairs
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/scratch/abernier/logs/scripts/%A.out
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
SOURCE_P0=1.014684793189  # pulse period
SOURCE_W50=60.5  # pulse width
PULSE_SEARCH_DOWNSAMP_FACTOR=8  # downsampling factor to apply to FDF_arr before pulse detection
SNR_CUTOFF=10  # S/N cutoff to apply when finding peaks in FDF

# Search results info
RESULTS_DIR=/scratch/abernier/pulsar_data/search_results/${SOURCE}/${OBS_NIGHT}
SETTINGS=timeavg391_nfiles300_allfiles_RFI1.7mean1.3std
PARAM=start_file_ind  # Specify which parameter was varied in the search results to plot
PARAM_LABEL="Start File Index"  # Label for the x-axis corresponding to the varied parameter (e.g., 'Downsampling Factor', 'DM [pc/cm^3]')
BURST_LIMS="0"  # burst limits in time
DATA_FILES_UNIQUE=0  # 1 to plot stokes only once, 0 to plot stokes for each param value 

# Run plotting script
cd /home/abernier/scratch/rm-search-Bernier-2025/scripts
echo "plots.py ${RESULTS_DIR} ${SETTINGS} ${PARAM} ${SOURCE_P0} ${SOURCE_W50}\
      --pulse_search_downsamp_factor ${PULSE_SEARCH_DOWNSAMP_FACTOR} \
      --snr_cutoff ${SNR_CUTOFF} \
      --param_label \"${PARAM_LABEL}\" \
      --burst_lims ${BURST_LIMS} \
      --data_files_unique ${DATA_FILES_UNIQUE}"
python plots.py ${RESULTS_DIR} ${SETTINGS} ${PARAM} ${SOURCE_P0} ${SOURCE_W50}\
      --pulse_search_downsamp_factor ${PULSE_SEARCH_DOWNSAMP_FACTOR} \
      --snr_cutoff ${SNR_CUTOFF} \
      --param_label "${PARAM_LABEL}" \
      --burst_lims ${BURST_LIMS} \
      --data_files_unique ${DATA_FILES_UNIQUE}