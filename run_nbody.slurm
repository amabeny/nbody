#!/bin/bash
#SBATCH --job-name=nbody_gpu
#SBATCH --output=nbody_gpu_output.txt
#SBATCH --error=nbody_gpu_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --ntasks=1

module load cuda

echo "Running GPU N-Body simulation on $SLURMD_NODENAME"
./nbody_gpu
