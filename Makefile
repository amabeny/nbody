# Makefile

# Compiler and flags
NVCC := nvcc
CFLAGS := -O2

# File names
TARGET := nbody
SRC := nbody.cu
SLURM_SCRIPT := run_nbody.slurm

# Default target: build the binary
all: $(TARGET)

# Compile the CUDA source
$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $@ $^

# Submit the SLURM job
submit: $(TARGET)
	sbatch $(SLURM_SCRIPT)

# Clean the build
clean:
	rm -f $(TARGET) *.o *.txt *.out
