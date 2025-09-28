# Basic CUDA Programming Exercises

## Exercise 1: Element-wise Operations

**Objective**: Implement various element-wise operations on arrays.

**Tasks**:
1. Implement element-wise multiplication of two arrays
2. Implement element-wise square root operation
3. Add error checking and timing
4. Compare GPU and CPU performance

**Template**:
```c
__global__ void elementMul(float* A, float* B, float* C, int n) {
    // TODO: Implement element-wise multiplication
}

__global__ void elementSqrt(float* input, float* output, int n) {
    // TODO: Implement element-wise square root
}
```

**Expected Learning**: Thread indexing, basic kernel structure, performance comparison.

---

## Exercise 2: Image Processing Basics

**Objective**: Apply simple filters to grayscale images.

**Tasks**:
1. Implement brightness adjustment (add constant to each pixel)
2. Implement contrast adjustment (multiply each pixel by constant)
3. Handle boundary conditions properly
4. Process images of different sizes

**Template**:
```c
__global__ void adjustBrightness(unsigned char* input, unsigned char* output, 
                                int width, int height, int brightness) {
    // TODO: Implement brightness adjustment
    // Remember to clamp values to [0, 255]
}
```

**Expected Learning**: 2D thread indexing, boundary handling, data clamping.

---

## Exercise 3: Dot Product

**Objective**: Compute dot product of two large vectors.

**Tasks**:
1. Implement a basic dot product kernel
2. Use shared memory for optimization
3. Handle arrays that don't fit exactly in block sizes
4. Implement recursive reduction for final sum

**Template**:
```c
__global__ void dotProduct(float* A, float* B, float* partialSums, int n) {
    extern __shared__ float sdata[];
    // TODO: Implement dot product with reduction
}
```

**Expected Learning**: Reduction patterns, shared memory usage, synchronization.

---

## Exercise 4: Histogram Calculation

**Objective**: Compute histogram of pixel intensities in an image.

**Tasks**:
1. Basic histogram kernel (may have race conditions)
2. Use atomic operations to handle conflicts
3. Optimize with shared memory privatization
4. Handle different bin counts

**Template**:
```c
__global__ void histogramBasic(unsigned char* input, int* histogram, int n) {
    // TODO: Basic histogram (watch for race conditions)
}

__global__ void histogramAtomic(unsigned char* input, int* histogram, int n) {
    // TODO: Use atomicAdd to avoid race conditions
}
```

**Expected Learning**: Race conditions, atomic operations, performance trade-offs.

---

## Exercise 5: Matrix Transpose

**Objective**: Efficiently transpose a matrix using shared memory.

**Tasks**:
1. Implement naive transpose
2. Implement coalesced memory access version
3. Use shared memory to avoid bank conflicts
4. Compare performance of different approaches

**Template**:
```c
__global__ void transposeNaive(float* input, float* output, int width, int height) {
    // TODO: Simple transpose (may have poor memory access patterns)
}

__global__ void transposeCoalesced(float* input, float* output, int width, int height) {
    // TODO: Optimized transpose with better memory access
}
```

**Expected Learning**: Memory coalescing, shared memory bank conflicts, optimization.

---

## Solutions

Solutions are provided in the `solutions/` directory. Try to implement the exercises yourself before looking at the solutions.

### Compilation Instructions

```bash
# Compile individual exercises
nvcc -O3 exercise1.cu -o exercise1
nvcc -O3 exercise2.cu -o exercise2
# ... etc

# Or use the provided Makefile
make all
```

### Testing

Each exercise should include:
- Input validation
- Error checking
- Performance timing
- Correctness verification against CPU reference

### Performance Tips

1. **Memory Access**: Ensure coalesced memory access patterns
2. **Occupancy**: Balance thread count with resource usage
3. **Shared Memory**: Use for data reuse and communication
4. **Atomic Operations**: Use sparingly due to serialization
5. **Divergence**: Minimize branch divergence within warps

### Common Pitfalls

1. **Array Bounds**: Always check `if (i < n)` in kernels
2. **Race Conditions**: Be careful with shared data structures
3. **Synchronization**: Use `__syncthreads()` when needed
4. **Memory Leaks**: Always free allocated memory
5. **Error Checking**: Check CUDA API return values

---

## Next Steps

After completing these exercises, move on to:
- [Intermediate Exercises](../intermediate/README.md)
- Advanced memory optimization techniques
- Multi-GPU programming
- CUDA libraries (cuBLAS, cuFFT, etc.)