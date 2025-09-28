#include <stdio.h>
#include <cuda_runtime.h>

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

// Matrix multiplication kernel (naive version)
__global__ void matrixMulNaive(float* A, float* B, float* C, int width) {
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

// Matrix multiplication with shared memory
__global__ void matrixMulShared(float* A, float* B, float* C, int width) {
    const int TILE_SIZE = 16;
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (width + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tiles into shared memory
        if (row < width && tile * TILE_SIZE + tx < width)
            As[ty][tx] = A[row * width + tile * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (col < width && tile * TILE_SIZE + ty < width)
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * width + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < width && col < width) {
        C[row * width + col] = sum;
    }
}

// CPU matrix multiplication for reference
void matrixMulCPU(float* A, float* B, float* C, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += A[row * width + k] * B[k * width + col];
            }
            C[row * width + col] = sum;
        }
    }
}

void initializeMatrix(float* matrix, int width) {
    for (int i = 0; i < width * width; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

bool verifyResult(float* C_gpu, float* C_cpu, int width, float tolerance = 1e-3) {
    for (int i = 0; i < width * width; i++) {
        if (fabs(C_gpu[i] - C_cpu[i]) > tolerance) {
            printf("Mismatch at index %d: GPU=%.6f, CPU=%.6f\n", 
                   i, C_gpu[i], C_cpu[i]);
            return false;
        }
    }
    return true;
}

int main() {
    const int WIDTH = 1024;
    size_t size = WIDTH * WIDTH * sizeof(float);
    
    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C_naive = (float*)malloc(size);
    float* h_C_shared = (float*)malloc(size);
    float* h_C_cpu = (float*)malloc(size);
    
    // Initialize matrices
    printf("Initializing %dx%d matrices...\n", WIDTH, WIDTH);
    initializeMatrix(h_A, WIDTH);
    initializeMatrix(h_B, WIDTH);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // Setup execution configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, 
                  (WIDTH + blockSize.y - 1) / blockSize.y);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Test naive implementation
    printf("Running naive matrix multiplication...\n");
    CUDA_CHECK(cudaEventRecord(start));
    matrixMulNaive<<<gridSize, blockSize>>>(d_A, d_B, d_C, WIDTH);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float naive_time;
    CUDA_CHECK(cudaEventElapsedTime(&naive_time, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, size, cudaMemcpyDeviceToHost));
    
    // Test shared memory implementation
    printf("Running shared memory matrix multiplication...\n");
    CUDA_CHECK(cudaEventRecord(start));
    matrixMulShared<<<gridSize, blockSize>>>(d_A, d_B, d_C, WIDTH);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float shared_time;
    CUDA_CHECK(cudaEventElapsedTime(&shared_time, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C_shared, d_C, size, cudaMemcpyDeviceToHost));
    
    // CPU reference (for small matrices only)
    if (WIDTH <= 512) {
        printf("Running CPU reference...\n");
        clock_t cpu_start = clock();
        matrixMulCPU(h_A, h_B, h_C_cpu, WIDTH);
        clock_t cpu_end = clock();
        float cpu_time = ((float)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000;
        
        // Verify results
        if (verifyResult(h_C_naive, h_C_cpu, WIDTH)) {
            printf("✓ Naive implementation verified\n");
        } else {
            printf("✗ Naive implementation failed verification\n");
        }
        
        if (verifyResult(h_C_shared, h_C_cpu, WIDTH)) {
            printf("✓ Shared memory implementation verified\n");
        } else {
            printf("✗ Shared memory implementation failed verification\n");
        }
        
        printf("\nPerformance Results:\n");
        printf("CPU time: %.2f ms\n", cpu_time);
        printf("GPU naive time: %.2f ms (%.2fx speedup)\n", 
               naive_time, cpu_time / naive_time);
        printf("GPU shared time: %.2f ms (%.2fx speedup)\n", 
               shared_time, cpu_time / shared_time);
    } else {
        printf("\nPerformance Results (no CPU verification for large matrices):\n");
        printf("GPU naive time: %.2f ms\n", naive_time);
        printf("GPU shared time: %.2f ms\n", shared_time);
    }
    
    printf("Shared memory speedup over naive: %.2fx\n", 
           naive_time / shared_time);
    
    // Calculate GFLOPS
    double gflops = (2.0 * WIDTH * WIDTH * WIDTH) / 1e9;
    printf("Naive GFLOPS: %.2f\n", gflops / (naive_time / 1000));
    printf("Shared GFLOPS: %.2f\n", gflops / (shared_time / 1000));
    
    // Clean up
    free(h_A); free(h_B); free(h_C_naive); free(h_C_shared); free(h_C_cpu);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}