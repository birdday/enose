#!/bin/bash
#SBATCH --job-name=RASPA_sims
#SBATCH --output=system.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-24:00:00

# If you wish to use scavenger partition, uncomment these lines.
# #SBATCH --partition=scavenger
# #SBATCH --requeue

# Use path to your virtual environment here.
source ~/venv/sensor_array_simulations/bin/activate

cp -pR $SLURM_SUBMIT_DIR/* $SLURM_SCRATCH
cd $SLURM_SCRATCH

run_on_exit(){
  cp -pR $SLURM_SCRATCH/Output/System_0/* $SLURM_SUBMIT_DIR
  cp -pR $SLURM_SCRATCH/*.out $SLURM_SUBMIT_DIR
}
trap run_on_exit EXIT

# If you call input something other than 'simulation.input', change this here.
simulate simulation.input
