#!/bin/bash
#SBATCH --job-name=nbody_cuda
#SBATCH --output=nbody_output.txt
#SBATCH --error=nbody_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --mem=4G

module load cuda

echo "Running CUDA N-body Simulation"
./nbody 1000 100
