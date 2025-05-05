CXXFLAGS=-O3
NVCC = nvcc

ARCH = -arch=sm_61
CFLAGS = -O2 -std=c++11

TARGET = nbody
SRC = nbody.cu

nbody: nbody.cpp
	g++ -O3 nbody.cpp -o nbody

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(ARCH) $(CFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)

solar.out: nbody
	date
	./nbody planet 200 5000000 10000 > solar.out # maybe a minutes
	date

solar.pdf: solar.out
	python3 plot.py solar.out solar.pdf 1000 

random.out: nbody
	date
	./nbody 1000 1 10000 100 > random.out # maybe 5 minutes
	date
