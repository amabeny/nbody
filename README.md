# nbody

## Compilation

Load the CUDA module:
    module load cuda/12.4

Then run:
    make

## Usage

Interactive test:
    ./nbody number 0.01 10 5 128

SLURM job:
    sbatch nbody.slurm

## Arguments
- Input type: `number`, `planet`, or filename
- Time step size (dt)
- Number of time steps
- Output interval (e.g., every 5 steps)
- CUDA block size

## Output
Each line starts with particle count followed by mass, position, velocity, and force per particle in tab-separated format.
