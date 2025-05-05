#include <iostream>
#include <cmath>
#include <cstdlib>
#include <iomanip> // For output formatting
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define G 6.67430e-11
// Define DT as a double for consistency
#define DT 0.01

struct Body {
    double x, y, z;
    double vx, vy, vz;
    double mass;
    double fx, fy, fz;
};

__global__ void update_positions(Body* bodies, int n, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double local_fx = 0.0;
    double local_fy = 0.0;
    double local_fz = 0.0;

    for (int j = 0; j < n; j++) {
        if (i == j) continue;
        double dx = bodies[j].x - bodies[i].x;
        double dy = bodies[j].y - bodies[i].y;
        double dz = bodies[j].z - bodies[i].z;
        double dist_sqr = dx * dx + dy * dy + dz * dz + 1e-9;
        double dist = sqrt(dist_sqr);
        double force = G * bodies[i].mass * bodies[j].mass / dist_sqr;

        local_fx += force * dx / dist;
        local_fy += force * dy / dist;
        local_fz += force * dz / dist;
    }

    bodies[i].fx = local_fx;
    bodies[i].fy = local_fy;
    bodies[i].fz = local_fz;

    bodies[i].vx += local_fx / bodies[i].mass * dt;
    bodies[i].vy += local_fy / bodies[i].mass * dt;
    bodies[i].vz += local_fz / bodies[i].mass * dt;

    bodies[i].x += bodies[i].vx * dt;
    bodies[i].y += bodies[i].vy * dt;
    bodies[i].z += bodies[i].vz * dt;
}

void initialize_bodies(Body* h_bodies, int n) {
    for (int i = 0; i < n; i++) {
        h_bodies[i].x = static_cast<double>(rand()) / RAND_MAX * 100.0;
        h_bodies[i].y = static_cast<double>(rand()) / RAND_MAX * 100.0;
        h_bodies[i].z = static_cast<double>(rand()) / RAND_MAX * 100.0;
        h_bodies[i].vx = 0.0;
        h_bodies[i].vy = 0.0;
        h_bodies[i].vz = 0.0;
        h_bodies[i].mass = static_cast<double>(rand()) / RAND_MAX * 1e5 + 1e3;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <number_of_bodies> <number_of_steps>" << std::endl;
        return 1;
    }

    int n = atoi(argv[1]);
    int steps = atoi(argv[2]);

    Body* h_bodies = new Body[n]; // Use new instead of malloc
    initialize_bodies(h_bodies, n);

    Body* d_bodies;
    cudaMalloc(&d_bodies, n * sizeof(Body));
    cudaMemcpy(d_bodies, h_bodies, n * sizeof(Body), cudaMemcpyHostToDevice);

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Record start time
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    for (int t = 0; t < steps; t++) {
        update_positions<<<numBlocks, BLOCK_SIZE>>>(d_bodies, n, DT);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "GPU Execution Time: " << milliseconds / 1000.0 << " seconds for " << n << " bodies and " << steps << " steps\n";

    cudaMemcpy(h_bodies, d_bodies, n * sizeof(Body), cudaMemcpyDeviceToHost);

    // Output first 5 bodies (formatted as in CPU output)
    std::cout << std::fixed << std::setprecision(10);
    std::cout << n << "\t";
    for (int i = 0; i < std::min(5, n); i++) { // Output up to 5 bodies
        std::cout << h_bodies[i].mass << "\t"
                  << h_bodies[i].x << "\t" << h_bodies[i].y << "\t" << h_bodies[i].z << "\t"
                  << h_bodies[i].vx << "\t" << h_bodies[i].vy << "\t" << h_bodies[i].vz << "\t"
                  << h_bodies[i].fx << "\t" << h_bodies[i].fy << "\t" << h_bodies[i].fz << "\t";
    }
    std::cout << std::endl;

    cudaFree(d_bodies);
    delete[] h_bodies; // Use delete[] for arrays allocated with new
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}
