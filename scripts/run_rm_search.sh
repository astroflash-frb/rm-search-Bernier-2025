#!/bin/bash
#SBATCH --job-name=rm_search
#SBATCH --account=def-istairs
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-29%5
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


# Set up parameter to vary
INDEX=${SLURM_ARRAY_TASK_ID}
STEP_LIST=(160 460 760 1060 1360 1660 1960 2260 2560 2860 3160 \
           3460 3760 4060 4360 4660 4960 5260 5560 5860 6160 6460 \
           6760 7060 7360 7660 7960 8260 8560 8860)  #(1 4 8 16 32 48 64 96 128 192 256)  # number of channels
PARAM_VARY=start_file_ind  # name of parameter being varied (e.g. "search_downsamp_factor")
# Things to update when changing the parameter to vary:
# 1) Update the STEP_LIST with the desired values of the parameter to vary 
# 2) update PARAM_VARY with the name of the parameter being varied (e.g. "search_downsamp_factor")
# 3) update the correct parameter variable below to use ${STEP_LIST[$INDEX]}
# 4) update the parameter variable/value in SAVE_FILE after PARAM_VARY
# 5) if changing the number of parameters, update the array job range above


# Source info/properties + input path
SOURCE=2022_CHIME_B2111+46
SOURCE_DIR=/scratch/abernier/pulsar_data/${SOURCE}
OBS_NIGHT=20220826T064420Z
DM=141.26  # pc/cm**3
RM=-218.70  # rad/m**2
DELAY=-2.0  # ns

# Data parameters
START_FILE_IND=${STEP_LIST[$INDEX]}
NFILES=300
REF_FREQ=600  # MHz
NPIXELS_TO_AVG=391  # number of pixels to average together in time when reading in the data (391 = 1ms time resolution)
SLICE_BURST_FLAG=0  # set to 1 to slice the data around the highest SNR burst, 0 to use the full data read into the array

# RM search parameters
PHI_MAX=1500
DPHI_SCALING=0.05
RFI_MEAN_THRESHOLD=1.7
RFI_STD_THRESHOLD=1.3
SEARCH_DOWNSAMP_FACTOR=1
COMPUTE_CROSS_CORR_FLAG=0

# Simulated burst parameters
SIM_FLAG=0  # Set to 1 to inject simulated bursts, 0 to run without
SIM_NUM_BURSTS=2
SIM_ARRIVAL_TIMES="2. 6."  # Need to be contained within the time range of the input data
SIM_BURST_WIDTHS="0.04 0.01"  # Widths of simulated bursts in seconds (or same time unit as data)
SIM_SNR="5 10"  # SNR of simulated bursts to inject (relative to noise in real timeseries data)
SIM_RM="-120 500"  # RM of simulated bursts to inject (should be within phi_max)

# Output path
STOKES_SETTINGS=timeavg${NPIXELS_TO_AVG}_nfiles${NFILES}_allfiles_RFI${RFI_MEAN_THRESHOLD}mean${RFI_STD_THRESHOLD}std
SUBDIR=${PARAM_VARY}_dm141
OUTDIR=/scratch/abernier/results_data/${SOURCE}/${OBS_NIGHT}/${STOKES_SETTINGS}/${SUBDIR}  # output directory
mkdir -p "$OUTDIR"  # Create output directory if it doesn't exist
SAVE_FILE=${OUTDIR}/${PARAM_VARY}_${START_FILE_IND}.npz  # Final save file

# Run RM Search
cd /home/abernier/scratch/rm-search-Bernier-2025/scripts  # go to script location
echo "01_rm_search.py ${SOURCE_DIR} ${OBS_NIGHT} ${SAVE_FILE}\
      --true_dm_pc_cm3 ${DM} \
      --true_rms_rad_m2 ${RM} \
      --delay_ns ${DELAY} \
      --start_file_ind ${START_FILE_IND} \
      --nfiles ${NFILES} \
      --ref_freq ${REF_FREQ} \
      --npixels_to_avg ${NPIXELS_TO_AVG} \
      --slice_burst_flag ${SLICE_BURST_FLAG} \
      --phi_max ${PHI_MAX} \
      --dphi_scaling ${DPHI_SCALING} \
      --rfi_mean_threshold ${RFI_MEAN_THRESHOLD} \
      --rfi_std_threshold ${RFI_STD_THRESHOLD} \
      --search_downsamp_factor ${SEARCH_DOWNSAMP_FACTOR} \
      --compute_cross_corr_flag ${COMPUTE_CROSS_CORR_FLAG} \
      --sim_flag ${SIM_FLAG} \
      --sim_num_bursts ${SIM_NUM_BURSTS} \
      --sim_arrival_times ${SIM_TIMES} \
      --sim_burst_widths ${SIM_BURST_WIDTHS} \
      --sim_snr ${SIM_SNR} \
      --sim_rm ${SIM_RM}"

python 01_rm_search.py ${SOURCE_DIR} ${OBS_NIGHT} ${SAVE_FILE}\
      --true_dm_pc_cm3 ${DM} \
      --true_rms_rad_m2 ${RM} \
      --delay_ns ${DELAY} \
      --start_file_ind ${START_FILE_IND} \
      --nfiles ${NFILES} \
      --ref_freq ${REF_FREQ} \
      --npixels_to_avg ${NPIXELS_TO_AVG} \
      --slice_burst_flag ${SLICE_BURST_FLAG} \
      --phi_max ${PHI_MAX} \
      --dphi_scaling ${DPHI_SCALING} \
      --rfi_mean_threshold ${RFI_MEAN_THRESHOLD} \
      --rfi_std_threshold ${RFI_STD_THRESHOLD} \
      --search_downsamp_factor ${SEARCH_DOWNSAMP_FACTOR} \
      --compute_cross_corr_flag ${COMPUTE_CROSS_CORR_FLAG} \
      --sim_flag ${SIM_FLAG} \
      --sim_num_bursts ${SIM_NUM_BURSTS} \
      --sim_arrival_times ${SIM_ARRIVAL_TIMES} \
      --sim_burst_widths ${SIM_BURST_WIDTHS} \
      --sim_snr ${SIM_SNR} \
      --sim_rm ${SIM_RM}