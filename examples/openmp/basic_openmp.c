#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

// Vector addition using OpenMP
void vectorAddOpenMP(float* a, float* b, float* c, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// Matrix multiplication using OpenMP
void matrixMultiplyOpenMP(float* A, float* B, float* C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// Parallel reduction using OpenMP
float reductionOpenMP(float* data, int n) {
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

int main() {
    const int N = 1024;
    const int size = N * sizeof(float);
    
    // Allocate memory
    float* a = (float*)malloc(size);
    float* b = (float*)malloc(size);
    float* c = (float*)malloc(size);
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }
    
    // Display OpenMP information
    printf("OpenMP Demo\n");
    printf("Max threads available: %d\n", omp_get_max_threads());
    printf("Number of processors: %d\n", omp_get_num_procs());
    
    // Vector addition
    double start = omp_get_wtime();
    vectorAddOpenMP(a, b, c, N);
    double end = omp_get_wtime();
    printf("Vector addition time: %.6f seconds\n", end - start);
    
    // Verify result
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(c[i] - (a[i] + b[i])) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("Vector addition result: %s\n", correct ? "CORRECT" : "INCORRECT");
    
    // Matrix multiplication
    float* A = (float*)malloc(N * N * sizeof(float));
    float* B = (float*)malloc(N * N * sizeof(float));
    float* C = (float*)malloc(N * N * sizeof(float));
    
    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }
    
    start = omp_get_wtime();
    matrixMultiplyOpenMP(A, B, C, N);
    end = omp_get_wtime();
    printf("Matrix multiplication time: %.6f seconds\n", end - start);
    
    // Reduction
    start = omp_get_wtime();
    float sum = reductionOpenMP(a, N);
    end = omp_get_wtime();
    printf("Reduction result: %.2f (time: %.6f seconds)\n", sum, end - start);
    
    // Clean up
    free(a); free(b); free(c);
    free(A); free(B); free(C);
    
    return 0;
}