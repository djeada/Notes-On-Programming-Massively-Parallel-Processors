# CUDA Programming Basics

## Overview

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model. It enables developers to use NVIDIA GPUs for general-purpose computing beyond graphics rendering.

## CUDA Programming Model

### Host and Device
- **Host**: The CPU and its memory (system RAM)
- **Device**: The GPU and its memory (VRAM)
- **Heterogeneous**: Applications execute on both host and device

### Kernel Functions
- Functions that execute on the GPU device
- Called from host code
- Executed by many threads in parallel

```c
__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
```

### Function Qualifiers

#### `__global__`
- Kernel functions executed on device
- Called from host (or device with dynamic parallelism)
- Must return `void`

#### `__device__`
- Functions executed on device
- Called from device code only
- Can return any type

#### `__host__`
- Functions executed on host
- Default qualifier for regular C++ functions
- Can be combined with `__device__`

## Thread Hierarchy

### Thread Organization
```
Grid
├── Block (0,0)    Block (1,0)    Block (2,0)
├── Block (0,1)    Block (1,1)    Block (2,1)
└── Block (0,2)    Block (1,2)    Block (2,2)

Each Block contains:
Thread (0,0)  Thread (1,0)  Thread (2,0)
Thread (0,1)  Thread (1,1)  Thread (2,1)
Thread (0,2)  Thread (1,2)  Thread (2,2)
```

### Built-in Variables

#### Grid Dimensions
- `gridDim.x`, `gridDim.y`, `gridDim.z`: Grid dimensions

#### Block Dimensions  
- `blockDim.x`, `blockDim.y`, `blockDim.z`: Block dimensions

#### Thread Indices
- `threadIdx.x`, `threadIdx.y`, `threadIdx.z`: Thread index within block
- `blockIdx.x`, `blockIdx.y`, `blockIdx.z`: Block index within grid

### Thread ID Calculation
```c
// 1D grid and blocks
int tid = blockIdx.x * blockDim.x + threadIdx.x;

// 2D grid and blocks
int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
```

## Memory Management

### Memory Allocation
```c
// Host memory
float* h_A = (float*)malloc(size);

// Device memory
float* d_A;
cudaMalloc((void**)&d_A, size);

// Free memory
free(h_A);
cudaFree(d_A);
```

### Memory Transfer
```c
// Host to device
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

// Device to host  
cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

// Device to device
cudaMemcpy(d_B, d_A, size, cudaMemcpyDeviceToDevice);
```

### Memory Types

#### Global Memory
- Largest memory space (GB)
- High latency (200-800 cycles)
- Accessible by all threads

#### Shared Memory
- Per-block memory (KB)
- Low latency (~1-30 cycles)
- Enables thread cooperation

```c
__global__ void kernel() {
    __shared__ float shared_data[256];
    // ... use shared memory
}
```

#### Registers
- Per-thread memory
- Fastest access (no latency)
- Limited quantity per thread

#### Constant Memory
- Read-only from device
- Cached for broadcast access
- 64 KB total

```c
__constant__ float const_data[1000];
```

## Kernel Launch Configuration

### Execution Configuration
```c
// 1D launch
kernel<<<numBlocks, blockSize>>>(args...);

// 2D launch  
dim3 grid(gridWidth, gridHeight);
dim3 block(blockWidth, blockHeight);
kernel<<<grid, block>>>(args...);

// With shared memory
kernel<<<grid, block, sharedMemSize>>>(args...);
```

### Block Size Considerations
- **Multiples of 32**: Warp size is 32 threads
- **Occupancy**: Balance threads and resources
- **Common sizes**: 128, 256, 512, 1024

## Error Checking

### CUDA Error Handling
```c
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Usage
CUDA_CHECK(cudaMalloc((void**)&d_A, size));
```

### Synchronization and Error Checking
```c
// Wait for kernel to finish
cudaDeviceSynchronize();

// Check for kernel launch errors
cudaError_t error = cudaGetLastError();
if (error != cudaSuccess) {
    printf("Kernel launch error: %s\n", cudaGetErrorString(error));
}
```

## Complete Example: Vector Addition

```c
#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define THREADS_PER_BLOCK 256

__global__ void vectorAdd(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    
    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Verify result
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            printf("Error at index %d\n", i);
            break;
        }
    }
    printf("Vector addition completed successfully\n");
    
    // Free memory
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}
```

## Device Properties

### Querying Device Information
```c
int deviceCount;
cudaGetDeviceCount(&deviceCount);

for (int device = 0; device < deviceCount; device++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("Device %d: %s\n", device, prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Global memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
    printf("Shared memory per block: %d KB\n", prop.sharedMemPerBlock / 1024);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
}
```

## Compilation and Execution

### NVCC Compiler
```bash
# Basic compilation
nvcc vector_add.cu -o vector_add

# With optimization
nvcc -O3 vector_add.cu -o vector_add

# Specify compute capability
nvcc -arch=sm_75 vector_add.cu -o vector_add

# Debug symbols
nvcc -g -G vector_add.cu -o vector_add
```

### Compute Capability
- **Architecture Version**: e.g., sm_75 for Turing
- **Feature Support**: Determines available GPU features
- **Optimization**: Compiler optimizations for specific architecture

## Performance Considerations

### Memory Access Patterns
```c
// Coalesced (efficient)
__global__ void coalesced(float* data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] = tid; // Sequential access
}

// Strided (less efficient)
__global__ void strided(float* data, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid * stride] = tid; // Strided access
}
```

### Thread Divergence
```c
// Divergent (inefficient)
__global__ void divergent(float* data, int* condition) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (condition[tid] > 0) {
        data[tid] *= 2.0f; // Some threads execute this
    } else {
        data[tid] *= 0.5f; // Others execute this
    }
}

// Predicated (better)
__global__ void predicated(float* data, int* condition) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float multiplier = (condition[tid] > 0) ? 2.0f : 0.5f;
    data[tid] *= multiplier; // All threads execute same path
}
```

## Debugging and Profiling

### CUDA-GDB
```bash
# Compile with debug info
nvcc -g -G program.cu -o program

# Debug
cuda-gdb ./program
```

### NVIDIA Nsight Systems
```bash
# Profile application
nsys profile --trace=cuda,nvtx ./program

# Generate report
nsys stats report.nsys-rep
```

## Common Patterns

### Reduction
```c
__global__ void reduction(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data to shared memory
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

### Matrix Multiplication (Basic)
```c
__global__ void matrixMul(float* A, float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}
```

## Summary

CUDA programming fundamentals include:

1. **Programming Model**: Host-device execution with kernel functions
2. **Thread Hierarchy**: Grid → Blocks → Threads organization  
3. **Memory Management**: Allocation, transfer, and different memory types
4. **Kernel Launch**: Execution configuration and parameter passing
5. **Error Handling**: Proper error checking and debugging
6. **Performance**: Memory access patterns and thread divergence
7. **Common Patterns**: Reduction, matrix operations, data transformations

Understanding these basics enables you to write efficient parallel programs on NVIDIA GPUs.

## Next Steps

Continue to [Memory Hierarchy](../04_memory_hierarchy/README.md) to learn advanced memory optimization techniques for high-performance CUDA applications.