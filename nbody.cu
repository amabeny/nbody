// nbody.cu
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define G 6.67430e-11

struct Body {
    double x, y, z;
    double vx, vy, vz;
    double mass;
    double fx, fy, fz; // To store forces if needed for output
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

    bodies[i].fx = local_fx; // Store forces (optional, for output)
    bodies[i].fy = local_fy;
    bodies[i].fz = local_fz;

    bodies[i].vx += local_fx / bodies[i].mass * dt;
    bodies[i].vy += local_fy / bodies[i].mass * dt;
    bodies[i].vz += local_fz / bodies[i].mass * dt;

    bodies[i].x += bodies[i].vx * dt;
    bodies[i].y += bodies[i].vy * dt;
    bodies[i].z += bodies[i].vz * dt;
    }
}

int main(int argc, char** argv) {
    int n = 1000; // default
    int steps = 100;

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) steps = atoi(argv[2]);

    Body* h_bodies = (Body*)malloc(n * sizeof(Body));
    initialize_bodies(h_bodies, n);

    Body* d_bodies;
    cudaMalloc(&d_bodies, n * sizeof(Body));
    cudaMemcpy(d_bodies, h_bodies, n * sizeof(Body), cudaMemcpyHostToDevice);

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < steps; t++) {
        update_positions<<<numBlocks, BLOCK_SIZE>>>(d_bodies, n);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_bodies, d_bodies, n * sizeof(Body), cudaMemcpyDeviceToHost);

    std::cout << "Final positions:\n";
    for (int i = 0; i < 5; i++) {
        std::cout << "Body " << i << ": (" << h_bodies[i].x << ", " << h_bodies[i].y << ")\n";
    }

    cudaFree(d_bodies);
    free(h_bodies);
    return 0;
}
