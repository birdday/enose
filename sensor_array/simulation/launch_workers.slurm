#!/bin/bash
#SBATCH --job-name=50mofs_10bar
#SBATCH --output=50mofs_10bar.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=0-24:00:00

# Example shell script for running job that runs off the Wilmerlab subjobserver.
# $Revision: 1.0 $
# $Date:  2016-03-21 $
# $Author: paulboone $
# Accepts a parameter stay_alive if you don't want the worker to exit immediately after all jobs are complete
# Use like `qsub -v stay_alive=1`

# $(date +"%Y-%m-%d_%H-%M-%S") --> Way to call the date in str format.

echo JOB_ID: $SLURM_JOBID JOB_NAME: $SLURM_JOBNAME HOSTNAME: $SLURM_O_HOST
echo start_time: `date`

cp -pR $SLURM_SUBMIT_DIR/* $SLURM_SCRATCH
cd $SLURM_SCRATCH

run_on_exit(){
  cp -pR $SLURM_SCRATCH/output_* /ihome/cwilmer/brd84/raspa/co2-sensing-results/CO2_0p1bar
  cp -pR $SLURM_SCRATCH/logs* /ihome/cwilmer/brd84/raspa/co2-sensing-results/CO2_0p1bar
  cp -pR $SLURM_SCRATCH/*.out /ihome/cwilmer/brd84/raspa/co2-sensing-results/CO2_0p1bar
}
trap run_on_exit EXIT

sjs_launch_workers.sh $SLURM_NTASKS_PER_NODE $stay_alive
