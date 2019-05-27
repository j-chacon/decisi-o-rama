#!/bin/bash

## set the time
#SBATCH -t 00:10:00

## Declare the number of nodes and task per node
#SBATCH -N 20
#SBATCH --tasks-per-node 10

## Give a name 
#SBATCH --job-name=test
#SBATCH --output=res.txt

srun hostname
srun sleep 60

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=j.chaconhurtado@tudelft.nl