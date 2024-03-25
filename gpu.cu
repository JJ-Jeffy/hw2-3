#include "common.h"
#include <cuda.h>
#include <iostream>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#define NUM_THREADS 256

// static global variables 
int blks;  
int num_bins;
int tot_bins;
double bin_size;


int* particle_ids;  // array for sorted particles 
int* bin_ids;       // array containing the index of the first particle that is stored in it 
int* bin_counts;    // array containing the number of particles in each bin
int* particle_counter;     // counter for the number of particles in each bin

// Function to set an array to zeros 
__global__ void set_to_zero(int* arr, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Initialize the arrays to 0 
    if (tid < size) {
        arr[tid] = 0;
    }
}

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int* particle_ids, int* bin_ids, int* bin_counts, int num_bins, double bin_size) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    // set the acceleration to zero
    particles[tid].ax = particles[tid].ay = 0;
    // find the bin index for the particles 
    int part_x = particles[tid].x / bin_size;
    int part_y = particles[tid].y / bin_size;

    // loop over the neighboring bins 
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {

            // get the neighboring bin id 
            int neighbor_x = part_x + i;
            int neighbor_y = part_y + j;

            // iterate through all the particles in the neighboring bin
            if (neighbor_y >= 0 && neighbor_y < num_bins && neighbor_x >= 0 && neighbor_x < num_bins) {
                int index = neighbor_x + neighbor_y * num_bins;
                
                // print the index 
                // printf("Index: %d\n", index);

                int neighbor_bin_start = bin_ids[index];
                int neighbor_bin_end = neighbor_bin_start + bin_counts[index];

                for (int k = neighbor_bin_start; k < neighbor_bin_end; k++) {
                    int neighbor_id = particle_ids[k];
                    if (neighbor_id != tid){
                        apply_force_gpu(particles[tid], particles[neighbor_id]);
                    }
                }
            }
        }
    }

}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

// Function to update the number of particles per bin using atomicAdd 
__global__ void update_bin_counts_gpu(particle_t* parts, int num_parts, int* bin_counts, int num_bins, double bin_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    int part_x = parts[tid].x / bin_size;
    int part_y = parts[tid].y / bin_size;
    int index = part_x + part_y * num_bins;
    atomicAdd(&bin_counts[index], 1);
}

// Function to update particle_ids array 
__global__ void update_particle_ids_gpu(particle_t* parts, int num_parts, int* particle_ids, int* bin_ids, int* bin_counts, int* particle_counter, int num_bins, double bin_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    int part_x = parts[tid].x / bin_size;
    int part_y = parts[tid].y / bin_size;
    int index = part_x + part_y * num_bins;
    int index_start = bin_ids[index];
    int loc = atomicAdd(&particle_counter[index], 1);
    particle_ids[index_start + loc] = tid;
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    num_bins = ceil(size / cutoff);
    tot_bins = num_bins * num_bins;
    bin_size = size / num_bins;

    // Allocate memory for particle_ids, bin_ids, and bin_counts
    cudaMalloc((void**)&particle_ids, num_parts * sizeof(int));
    cudaMalloc((void**)&bin_ids, tot_bins * sizeof(int));
    cudaMalloc((void**)&bin_counts, tot_bins * sizeof(int));
    cudaMalloc((void**)&particle_counter, tot_bins * sizeof(int));

    // set everything to zero 
    set_to_zero<<<blks, NUM_THREADS>>>(bin_counts, tot_bins);
    set_to_zero<<<blks, NUM_THREADS>>>(particle_ids, num_parts);
    set_to_zero<<<blks, NUM_THREADS>>>(bin_ids, tot_bins);
    set_to_zero<<<blks, NUM_THREADS>>>(particle_counter, tot_bins);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function
    
    // set everything to zero 
    set_to_zero<<<blks, NUM_THREADS>>>(bin_counts, tot_bins);
    set_to_zero<<<blks, NUM_THREADS>>>(particle_ids, num_parts);
    set_to_zero<<<blks, NUM_THREADS>>>(bin_ids,tot_bins);
    set_to_zero<<<blks, NUM_THREADS>>>(particle_counter, tot_bins);

    // Update bin_counts and particle_ids
    update_bin_counts_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, bin_counts, num_bins, bin_size);

    // Perform exclusive scan on bin_counts
    thrust::inclusive_scan(thrust::device, bin_counts, bin_counts + tot_bins, bin_ids);

    // // Copy the result back to bin_counts
    // cudaMemcpy(bin_counts, bin_ids, num_bins * num_bins * sizeof(int), cudaMemcpyDeviceToDevice);

    // Update particle_ids
    update_particle_ids_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, particle_ids, bin_ids, bin_counts, particle_counter, num_bins, bin_size);

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, particle_ids, bin_ids, bin_counts, num_bins, bin_size);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}