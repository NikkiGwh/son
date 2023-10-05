#!/bin/bash

# Run one task on one node
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Make all cores available to our task
#SBATCH --cpus-per-task=2
# Allocate Memory
#SBATCH --mem-per-cpu=4G
# Use the 'all' partition
#SBATCH --partition=all
# Redirect output and error output
#SBATCH --output=job.log.out
#SBATCH --error=job.log.err

srun hostname
srun nproc
# Jetzt kommt der tatsächliche Aufruf. Dem koennen auch Argumente übergeben werden.
# Ebenfalls kann mit apptainer exec ein bestimmter Befehl ausgeführt werden
srun apptainer run nns.sif
srun echo "Experiment Done"