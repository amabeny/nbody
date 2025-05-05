#!/bin/bash
#SBATCH --job-name=gpu_bench
#SBATCH --partition=GPU
#SBATCH --time=00:30:00 # Adjust time as needed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=gpu_benchmarks_%j.out
#SBATCH --error=gpu_benchmarks_%j.err

module load cuda/12.4

steps=10
particles=(1000 10000 100000)
log_file="gpu_benchmarks.log"

echo "GPU Benchmarks (SLURM)" > "$log_file"

for p in "${particles[@]}"; do
    start_time=$(date +%s.%N)
    ./nbody_gpu "$p" "$steps" > "gpu_${p}_slurm.txt"
    end_time=$(date +%s.%N)
    elapsed_time=$(grep "GPU Execution Time" "gpu_${p}_slurm.txt" | awk '{print $4}')
    echo "$p particles (SLURM): $elapsed_time seconds" >> "$log_file"
    echo "GPU ($p particles, SLURM): Execution Time: $elapsed_time seconds (Job ID: $SLURM_JOB_ID)"
done

echo "GPU benchmarking complete (SLURM). Results in $log_file"
