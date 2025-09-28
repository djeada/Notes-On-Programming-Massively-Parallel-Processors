# Introduction to Parallel Computing

## Overview

Parallel computing is the simultaneous execution of multiple computational tasks to solve a problem faster than sequential execution. This chapter introduces fundamental concepts and motivations for parallel programming.

## Why Parallel Computing?

### The End of Dennard Scaling
- **Moore's Law**: Transistor density doubles approximately every two years
- **Dennard Scaling**: Power density remains roughly constant with technology scaling
- **The Problem**: Around 2005, Dennard scaling ended due to power wall and heat dissipation limits
- **The Solution**: Multi-core processors and specialized parallel architectures

### Performance Benefits
- **Speedup**: Reduction in execution time
- **Throughput**: Increased number of operations per unit time
- **Energy Efficiency**: Better performance per watt ratios

## Types of Parallelism

### 1. Task Parallelism
- Different tasks execute simultaneously on different processors
- Tasks may be independent or have dependencies
- Examples: Pipeline processing, producer-consumer patterns

### 2. Data Parallelism
- Same operation performed on different data elements simultaneously
- Most common form in GPU computing
- Examples: Vector operations, matrix multiplication, image processing

### 3. Instruction-Level Parallelism (ILP)
- Multiple instructions execute simultaneously within a single processor
- Handled automatically by modern CPUs
- Examples: Superscalar execution, out-of-order execution

## Flynn's Taxonomy

Classification of parallel architectures based on instruction and data streams:

### SISD (Single Instruction, Single Data)
- Traditional sequential computers
- One instruction operates on one data element at a time
- Example: Early CPUs

### SIMD (Single Instruction, Multiple Data)
- Same instruction operates on multiple data elements
- Foundation of vector processors and GPU computing
- Example: GPU warps, CPU SIMD instructions (SSE, AVX)

### MISD (Multiple Instruction, Single Data)
- Different instructions operate on the same data
- Rare in practice
- Example: Fault-tolerant systems with redundant processing

### MIMD (Multiple Instruction, Multiple Data)
- Different instructions operate on different data
- Most flexible parallel architecture
- Example: Multi-core CPUs, distributed systems

## Performance Metrics

### Speedup
```
Speedup = T_serial / T_parallel
```
Where:
- T_serial: Execution time on single processor
- T_parallel: Execution time on multiple processors

### Efficiency
```
Efficiency = Speedup / Number_of_Processors
```
- Measures utilization of parallel resources
- Ideal efficiency = 1.0 (100%)

### Scalability
- **Strong Scaling**: Fixed problem size, increasing number of processors
- **Weak Scaling**: Problem size increases proportionally with processors

## Amdahl's Law

**Statement**: The speedup of a program using multiple processors is limited by the sequential fraction of the program.

```
Speedup = 1 / (S + (1-S)/P)
```
Where:
- S: Sequential fraction (0 ≤ S ≤ 1)
- P: Number of processors
- (1-S): Parallel fraction

### Key Insights
1. **Sequential bottleneck**: Even small sequential portions limit speedup
2. **Diminishing returns**: Adding more processors yields less benefit
3. **Maximum speedup**: Limited by 1/S for infinite processors

### Example
If 10% of a program is sequential (S = 0.1):
- Maximum speedup = 1/0.1 = 10x
- With 100 processors: Speedup = 1/(0.1 + 0.9/100) ≈ 9.2x

## Gustafson's Law

**Alternative perspective**: For a fixed execution time, larger problems can utilize more processors effectively.

```
Scaled_Speedup = S + P(1-S)
```
Where:
- S: Sequential fraction in scaled problem
- P: Number of processors

### Key Insight
As problem size increases, parallel portion dominates, enabling better scalability.

## Parallel Computing Challenges

### 1. Load Balancing
- **Problem**: Uneven work distribution among processors
- **Solution**: Dynamic load balancing, work stealing

### 2. Communication Overhead
- **Problem**: Time spent moving data between processors
- **Solution**: Minimize communication, overlap with computation

### 3. Synchronization
- **Problem**: Coordinating parallel tasks
- **Solution**: Barriers, locks, atomic operations

### 4. Memory Consistency
- **Problem**: Ensuring consistent view of shared data
- **Solution**: Memory models, synchronization primitives

### 5. Race Conditions
- **Problem**: Non-deterministic behavior due to timing
- **Solution**: Proper synchronization, atomic operations

## Parallel Programming Models

### 1. Shared Memory
- Processors share global address space
- Communication through shared variables
- Examples: OpenMP, Pthreads

### 2. Message Passing
- Processors have private memory spaces
- Communication through explicit messages
- Examples: MPI, distributed computing

### 3. Data Parallel
- Same operation on different data elements
- High-level abstraction for parallel algorithms
- Examples: CUDA, OpenCL, parallel array languages

## Applications of Parallel Computing

### Scientific Computing
- Climate modeling
- Molecular dynamics
- Computational fluid dynamics
- N-body simulations

### Machine Learning
- Training neural networks
- Computer vision
- Natural language processing
- Recommendation systems

### Graphics and Visualization
- Real-time rendering
- Ray tracing
- Image and video processing
- Computer-aided design

### Big Data Analytics
- Data mining
- Database queries
- Search algorithms
- Graph processing

## Summary

Parallel computing is essential for modern high-performance applications. Key takeaways:

1. **Motivation**: Power constraints ended single-core performance scaling
2. **Types**: Task, data, and instruction-level parallelism
3. **Metrics**: Speedup, efficiency, and scalability measure performance
4. **Limits**: Amdahl's law shows sequential portions limit speedup
5. **Challenges**: Load balancing, communication, synchronization
6. **Models**: Shared memory, message passing, data parallel approaches
7. **Applications**: Scientific computing, ML, graphics, big data

Understanding these fundamentals is crucial for effective parallel programming and optimization.

## Next Steps

Continue to [GPU Architecture](../02_gpu_architecture/README.md) to learn about specialized parallel processors designed for massively parallel computations.