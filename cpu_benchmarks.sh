#!/bin/bash

# Benchmarking CPU code

dt=0.01
steps=10
printevery=0 # Only print the final state for faster benchmarking
log_file="cpu_benchmarks.log"

echo "CPU Benchmarks" > "$log_file"

particles=(1000 10000 100000)

for p in "${particles[@]}"; do
    start_time=$(date +%s.%N)
    ./nbody_cpu "$p" "$dt" "$steps" "$printevery" > "cpu_${p}.txt"
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc -l)
    echo "$p particles: $elapsed_time seconds" >> "$log_file"
    echo "CPU ($p particles): Execution Time: $elapsed_time seconds"
done

echo "CPU benchmarking complete. Results in $log_file"
