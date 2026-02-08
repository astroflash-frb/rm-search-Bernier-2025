#!/bin/bash
#SBATCH --account=def-istairs
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/abernier/logs/scripts/%A.out

# Load modules
cd ~/scratch
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2024b
echo "Modules loaded"

# Load python environment
source /home/abernier/envs/tril_env/bin/activate
echo "Environment loaded"


# RM search parameters
PHI_MAX=1500
DPHI_SCALING=0.05
RFI_MEAN_THRESHOLD=2
RFI_STD_THRESHOLD=1.3
RM_TIME_STEP=10

# Input data path
DATA_DIR=/scratch/abernier/pulsar_data/baseband_full_stokes/2022_CHIME_B2111+46
DATA_FILE=${DATA_DIR}/20220826T064420Z_bands456_timeavg391_300files_start160.npz

# Output paths
OUTDIR=/scratch/abernier/pulsar_data/search_results  # Output directory
mkdir -p "$OUTDIR"  # Create output directory if it doesn't exist
SAVE_FILE=${OUTDIR}/FinalTest.npz  # Final save file
# will need to change for array jobs (source_night_[param getting changed]_arraynum?? all metrics get saved anyway)


# Run RM Search
cd /home/abernier/scratch/rm-search-Bernier-2025
echo "rm_search.py ${DATA_FILE} ${SAVE_FILE}\
      --phi_max ${PHI_MAX} \
      --dphi_scaling ${DPHI_SCALING} \
      --rfi_mean_threshold ${RFI_MEAN_THRESHOLD} \
      --rfi_std_threshold ${RFI_STD_THRESHOLD} \
      --rm_time_step ${RM_TIME_STEP}"

python rm_search.py ${DATA_FILE} ${SAVE_FILE}\
      --phi_max ${PHI_MAX} \
      --dphi_scaling ${DPHI_SCALING} \
      --rfi_mean_threshold ${RFI_MEAN_THRESHOLD} \
      --rfi_std_threshold ${RFI_STD_THRESHOLD} \
      --rm_time_step ${RM_TIME_STEP}