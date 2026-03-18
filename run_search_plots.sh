#!/bin/bash
#SBATCH --job-name=search_plots
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
TIME_RES=391  # [391, 781, 1563, 3125, 6250, 12500, 25000] for 1ms to 64ms resolution
SETTINGS=20220826T064420Z_bands6_timeavg${TIME_RES}_300files_start160  # Specify which Stokes data was used in the search
PARAM=timestep  # Specify which parameter was varied in the search results to plot
RM_TRUE=-218.70

# Run Ploting Script
cd /home/abernier/scratch/rm-search-Bernier-2025
echo "search_plots.py ${SETTINGS} ${PARAM} --rm_true ${RM_TRUE}"

python search_plots.py ${SETTINGS} ${PARAM} --rm_true ${RM_TRUE}