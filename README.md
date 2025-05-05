# nbody

This project implements a parallel N-body simulation using CUDA on a GPU and a sequential version for the CPU.
## Repository Contents

* `nbody.cpp`: C++ source code for the CPU-based N-body simulation.
* `nbody.cu`: CUDA source code for the GPU-based N-body simulation.
* `Makefile`:  Builds the `nbody_cpu` and `nbody_gpu` executables.
* `README.md`: This file, providing instructions for compilation, usage, and benchmarking.
* `run_gpu_benchmarks.slurm`: SLURM script for running GPU benchmarks on Centaurus.
* `cpu_benchmarks.sh`: Bash script for running CPU benchmarks.
* `cpu_benchmarks.log`: Log file containing CPU benchmark results.
* `gpu_benchmarks.log`: Log file containing GPU benchmark results.

## Prerequisites

* A C++ compiler (e.g., g++)
* CUDA Toolkit (nvcc) - If working on a system with an NVIDIA GPU.
* SLURM - If running the GPU code on a cluster like Centaurus.

## Compilation

1.  **CUDA Module (Centaurus):** If you are on Centaurus, load the CUDA module:

    ```bash
    module load cuda/12.4
    ```

2.  **Build:** Compile the code using the provided `Makefile`:

    ```bash
    make
    ```

    This will create the executables `nbody_cpu` and `nbody_gpu`.

## Running the Simulation

### CPU

To run the CPU simulation:

```bash
./nbody_cpu <number_of_particles> <time_step> <number_of_steps> <print_every> > output.txt
