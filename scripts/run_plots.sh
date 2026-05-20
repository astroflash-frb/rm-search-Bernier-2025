#!/bin/bash
#SBATCH --job-name=search_plots
#SBATCH --account=def-istairs
#SBATCH --time=00:30:00
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

# Search results info
SOURCE=2022_CHIME_B2111+46
OBS_NIGHT=20220826T064420Z
RESULTS_DIR=/scratch/abernier/pulsar_data/search_results/${SOURCE}/${OBS_NIGHT}
SETTINGS=timeavg391_nfiles65_start160_RFI1.7mean1.3std
PARAM=true_dm_pc_cm3  # Specify which parameter was varied in the search results to plot
PARAM_LABEL="DM [pc/cm^3]"  # Label for the x-axis corresponding to the varied parameter (e.g., 'Downsampling Factor', 'DM [pc/cm^3]')
BURST_LIMS="1.10 1.15"  # to slice around burst for plots

# Run plotting script
cd /home/abernier/scratch/rm-search-Bernier-2025/scripts
echo "plots.py ${RESULTS_DIR} ${SETTINGS} ${PARAM} \
      --param_label \"${PARAM_LABEL}\" \
      --burst_lims ${BURST_LIMS}"
python plots.py ${RESULTS_DIR} ${SETTINGS} ${PARAM} \
      --param_label "${PARAM_LABEL}" \
      --burst_lims ${BURST_LIMS} #\
      #--data_files_unique