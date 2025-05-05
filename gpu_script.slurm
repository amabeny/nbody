#!/bin/bash

# benchmarking GPU code

steps=10
particles=(1000 10000 100000)
log_file="gpu_benchmarks.log"

echo "GPU Benchmarks" > "$log_file"

for p in "${particles[@]}"; do
    ./nbody_gpu "$p" "$steps" > "gpu_${p}.txt"
    execution_time=$(grep "GPU Execution Time" "gpu_${p}.txt" | awk '{print $4}') # Extract the time
    echo "$p particles: $execution_time seconds" >> "$log_file"
done

echo "GPU benchmarking complete. Results in $log_file"
