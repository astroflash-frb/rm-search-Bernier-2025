#!/bin/bash
#SBATCH --job-name=baseband_formatter
#SBATCH --account=def-istairs
#SBATCH --time=3:00:00
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


# Parameters
SOURCE_DIR=/scratch/abernier/pulsar_data/2022_CHIME_B2111+46
OBS_NIGHT=20220826T064420Z_CHIME_vdif
SAVE_DIR=/scratch/abernier/pulsar_data/baseband_full_stokes
START_FILE_IND=160
NFILES=300
FREQ_BANDS="6"  # Specify which frequency bands to process (e.g., "4 5 6" or "6")
NSAMPLES_PER_CHUNK=25000  # [391, 781, 1563, 3125, 6250, 12500, 25000] for 1ms to 64ms resolution


# Run baseband formatter
cd /home/abernier/scratch/rm-search-Bernier-2025
echo "baseband_formatter.py ${SOURCE_DIR} ${OBS_NIGHT} ${SAVE_DIR} ${FREQ_BANDS}\
      --start_file_ind ${START_FILE_IND} \
      --nfiles ${NFILES} \
      --num_time_samples ${NSAMPLES_PER_CHUNK}"

python baseband_formatter.py ${SOURCE_DIR} ${OBS_NIGHT} ${SAVE_DIR} ${FREQ_BANDS}\
      --start_file_ind ${START_FILE_IND} \
      --nfiles ${NFILES} \
      --num_time_samples ${NSAMPLES_PER_CHUNK}