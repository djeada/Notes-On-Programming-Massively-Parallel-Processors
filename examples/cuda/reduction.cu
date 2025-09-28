#include <stdio.h>
#include <cuda_runtime.h>
#include <cmath>

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

// Simple reduction kernel (inefficient - for demonstration)
__global__ void reductionSimple(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data to shared memory
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();
    
    // Reduction in shared memory (naive approach)
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Optimized reduction kernel
__global__ void reductionOptimized(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Load and add two elements per thread
    sdata[tid] = (i < n ? input[i] : 0) + (i + blockDim.x < n ? input[i + blockDim.x] : 0);
    __syncthreads();
    
    // Reduction in shared memory (optimized)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Warp-level reduction using shuffle instructions
__device__ float warpReduce(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Modern reduction with warp primitives
__global__ void reductionWarp(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Load and add two elements per thread
    float val = (i < n ? input[i] : 0) + (i + blockDim.x < n ? input[i + blockDim.x] : 0);
    
    // Warp-level reduction
    val = warpReduce(val);
    
    // Store warp results in shared memory
    if (tid % 32 == 0) {
        sdata[tid / 32] = val;
    }
    __syncthreads();
    
    // Final reduction of warp results
    if (tid < 32) {
        val = (tid < blockDim.x / 32) ? sdata[tid] : 0;
        val = warpReduce(val);
    }
    
    if (tid == 0) output[blockIdx.x] = val;
}

// CPU reduction for reference
float reductionCPU(float* input, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += input[i];
    }
    return sum;
}

// Recursive GPU reduction launcher
float reductionGPU(float* d_input, int n, int kernel_type = 2) {
    int threads = THREADS_PER_BLOCK;
    int blocks = (kernel_type == 2) ? (n + threads * 2 - 1) / (threads * 2) : (n + threads - 1) / threads;
    
    float* d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_output, blocks * sizeof(float)));
    
    // Launch appropriate kernel
    switch (kernel_type) {
        case 0:
            reductionSimple<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_output, n);
            break;
        case 1:
            reductionOptimized<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_output, n);
            break;
        case 2:
            reductionWarp<<<blocks, threads, (threads / 32) * sizeof(float)>>>(d_input, d_output, n);
            break;
    }
    
    CUDA_CHECK(cudaGetLastError());
    
    // If multiple blocks, recursively reduce
    if (blocks > 1) {
        float result = reductionGPU(d_output, blocks, kernel_type);
        CUDA_CHECK(cudaFree(d_output));
        return result;
    } else {
        float result;
        CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_output));
        return result;
    }
}

int main() {
    const int N = 16 * 1024 * 1024; // 16M elements
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float* h_input = (float*)malloc(size);
    
    // Initialize input data
    printf("Initializing %d elements...\n", N);
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f; // Simple case: all ones, sum should be N
    }
    
    // CPU computation for reference
    printf("Computing CPU reference...\n");
    clock_t cpu_start = clock();
    float cpu_result = reductionCPU(h_input, N);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    
    // Allocate device memory
    float* d_input;
    CUDA_CHECK(cudaMalloc((void**)&d_input, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const char* kernel_names[] = {"Simple", "Optimized", "Warp-level"};
    
    // Test different reduction kernels
    for (int kernel_type = 0; kernel_type < 3; kernel_type++) {
        printf("\nTesting %s reduction...\n", kernel_names[kernel_type]);
        
        CUDA_CHECK(cudaEventRecord(start));
        float gpu_result = reductionGPU(d_input, N, kernel_type);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float gpu_time;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
        gpu_time /= 1000.0f; // Convert to seconds
        
        // Verify result
        bool correct = fabs(gpu_result - cpu_result) < 1e-3;
        printf("Result: %.0f (CPU: %.0f) %s\n", 
               gpu_result, cpu_result, correct ? "✓" : "✗");
        
        if (correct) {
            printf("Time: %.4f s (%.2fx speedup over CPU)\n", 
                   gpu_time, cpu_time / gpu_time);
            
            // Calculate bandwidth
            double bandwidth = (size / gpu_time) / 1e9;
            printf("Bandwidth: %.2f GB/s\n", bandwidth);
        }
    }
    
    // Clean up
    free(h_input);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}