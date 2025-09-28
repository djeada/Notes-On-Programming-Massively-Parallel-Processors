#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 1024*1024  // 1M elements
#define THREADS_PER_BLOCK 256

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// CUDA kernel for vector addition
__global__ void vectorAdd(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// CPU version for comparison
void vectorAddCPU(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    float* h_C_ref = (float*)malloc(size); // CPU reference result
    
    // Initialize host arrays
    printf("Initializing arrays with %d elements...\n", N);
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }
    
    // CPU computation for reference
    clock_t start_cpu = clock();
    vectorAddCPU(h_A, h_B, h_C_ref, N);
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Copy data to device
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // Launch kernel
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    printf("Launching kernel with %d blocks of %d threads each\n", blocks, THREADS_PER_BLOCK);
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    
    // Wait for events to complete
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate timing
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
    gpu_time /= 1000.0f; // Convert to seconds
    
    // Verify result
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-5) {
            printf("Error at index %d: GPU=%.6f, CPU=%.6f\n", i, h_C[i], h_C_ref[i]);
            correct = false;
            break;
        }
    }
    
    if (correct) {
        printf("✓ Results match!\n");
    } else {
        printf("✗ Results do not match!\n");
    }
    
    // Print performance results
    printf("\nPerformance Results:\n");
    printf("CPU time: %.4f seconds\n", cpu_time);
    printf("GPU time: %.4f seconds\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    
    // Calculate bandwidth
    double bytes_transferred = 3.0 * size; // Read A, B; Write C
    double bandwidth_gpu = bytes_transferred / (gpu_time * 1e9);
    printf("GPU Bandwidth: %.2f GB/s\n", bandwidth_gpu);
    
    // Clean up
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}