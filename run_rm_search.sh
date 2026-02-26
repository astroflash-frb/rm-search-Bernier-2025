#!/bin/bash
#SBATCH --job-name=rm_search
#SBATCH --account=def-istairs
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-9%5
#
#SBATCH --output=/scratch/abernier/logs/scripts/_%j_%a.out
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

# Set up parameter to vary
INDEX=${SLURM_ARRAY_TASK_ID}
STEP_LIST=(1 3 5 10 15 20 30 50 100 150)
PARAM_VARY=timestep

# RM search parameters
PHI_MAX=1500
DPHI_SCALING=0.05
RFI_MEAN_THRESHOLD=2.
RFI_STD_THRESHOLD=1.3
SEARCH_TIME_STEP=${STEP_LIST[$INDEX]}  # Vary this parameter across array jobs

# Input data path
DATA_DIR=/scratch/abernier/pulsar_data/baseband_full_stokes/2022_CHIME_B2111+46
SETTINGS=bands456_timeavg391_300files_start160
DATA_FILE=${DATA_DIR}/20220826T064420Z_${SETTINGS}.npz

# Output paths
OUTDIR=/scratch/abernier/pulsar_data/search_results/20220826T064420Z_${SETTINGS}/${PARAM_VARY} # Output directory
mkdir -p "$OUTDIR"  # Create output directory if it doesn't exist
SAVE_FILE=${OUTDIR}/2022_CHIME_B2111+46_20220826T064420Z_${PARAM_VARY}${SEARCH_TIME_STEP}.npz  # Final save file

# Run RM Search
cd /home/abernier/scratch/rm-search-Bernier-2025
echo "rm_search.py ${DATA_FILE} ${SAVE_FILE}\
      --phi_max ${PHI_MAX} \
      --dphi_scaling ${DPHI_SCALING} \
      --rfi_mean_threshold ${RFI_MEAN_THRESHOLD} \
      --rfi_std_threshold ${RFI_STD_THRESHOLD} \
      --search_time_step ${SEARCH_TIME_STEP}"

python rm_search.py ${DATA_FILE} ${SAVE_FILE}\
      --phi_max ${PHI_MAX} \
      --dphi_scaling ${DPHI_SCALING} \
      --rfi_mean_threshold ${RFI_MEAN_THRESHOLD} \
      --rfi_std_threshold ${RFI_STD_THRESHOLD} \
      --search_time_step ${SEARCH_TIME_STEP}