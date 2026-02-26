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
SETTINGS=20220826T064420Z_bands456_timeavg391_300files_start160  # Specify which Stokes data was used in the search
PARAM=timestep  # Specify which parameter was varied in the search results to plot

# Run Ploting Script
cd /home/abernier/scratch/rm-search-Bernier-2025
echo "search_plots.py ${SETTINGS} ${PARAM}"

python search_plots.py ${SETTINGS} ${PARAM}