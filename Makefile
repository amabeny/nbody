# Makefile for N-Body Simulation (CPU and GPU)

# Compiler and flags
CXX = g++
NVCC = nvcc
CXXFLAGS = -O2 -std=c++11
NVCCFLAGS = -O2 -arch=sm_70  # adjust arch for your Centaurus GPU node

# Targets
TARGET_CPU = nbody_cpu
TARGET_GPU = nbody_gpu

# Source files
SRC_CPU = nbody.cpp
SRC_GPU = nbody.cu

# Rules
all: $(TARGET_CPU) $(TARGET_GPU)

$(TARGET_CPU): $(SRC_CPU)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(TARGET_GPU): $(SRC_GPU)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET_CPU) $(TARGET_GPU)
