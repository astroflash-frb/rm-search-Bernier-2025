#!/bin/bash
#SBATCH --account=def-istairs
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-7
#SBATCH --output=/scratch/abernier/logs/scripts/%A_%a.out






# SOURCE_DIR = '/scratch/abernier/pulsar_data/2022_CHIME_B2111+46'
# SAVE_DIR = '/scratch/abernier/pulsar_data/baseband_full_stokes'
# START_FILE_IND = 160
# NFILES = 300
# FREQ_BANDS = [4,5,6]
# NSAMPLES_PER_CHUNK = 100
# NIGHT = '20220826T064420Z_CHIME_vdif'