// nbody.cu
#include <iostream>
#include <cmath>
#include <cstdlib>

#define BLOCK_SIZE 256
#define G 6.67430e-11f
#define DT 0.01f

struct Body {
    float x, y;
    float vx, vy;
    float mass;
};

__global__ void update_positions(Body* bodies, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float fx = 0.0f;
    float fy = 0.0f;

    for (int j = 0; j < n; j++) {
        if (i == j) continue;
        float dx = bodies[j].x - bodies[i].x;
        float dy = bodies[j].y - bodies[i].y;
        float dist_sqr = dx * dx + dy * dy + 1e-9f;
        float force = G * bodies[i].mass * bodies[j].mass / dist_sqr;
        float dist = sqrtf(dist_sqr);
        fx += force * dx / dist;
        fy += force * dy / dist;
    }

    float ax = fx / bodies[i].mass;
    float ay = fy / bodies[i].mass;

    bodies[i].vx += ax * DT;
    bodies[i].vy += ay * DT;
    bodies[i].x += bodies[i].vx * DT;
    bodies[i].y += bodies[i].vy * DT;
}

void initialize_bodies(Body* bodies, int n) {
    for (int i = 0; i < n; i++) {
        bodies[i].x = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        bodies[i].y = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        bodies[i].vx = 0.0f;
        bodies[i].vy = 0.0f;
        bodies[i].mass = static_cast<float>(rand()) / RAND_MAX * 1e5f + 1e3f;
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
