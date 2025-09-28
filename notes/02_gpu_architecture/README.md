# GPU Architecture

## Overview

Graphics Processing Units (GPUs) have evolved from specialized graphics accelerators to general-purpose parallel processors capable of handling diverse computational workloads. This chapter explores GPU architecture fundamentals and design principles.

## Evolution of GPUs

### Graphics Pipeline Origins
- **Fixed-Function**: Early GPUs had hardwired graphics operations
- **Programmable Shaders**: Vertex and pixel shaders introduced programmability
- **Unified Architecture**: Single processor type handles all shader stages
- **GPGPU**: General-Purpose GPU computing for non-graphics applications

### Key Milestones
- **2006**: NVIDIA CUDA introduced programmable GPU computing
- **2008**: OpenCL standard for heterogeneous computing
- **2010s**: GPU computing becomes mainstream for HPC and AI
- **2020s**: AI/ML workloads drive GPU architecture evolution

## CPU vs GPU Design Philosophy

### CPU Design (Latency-Oriented)
- **Few Cores**: 4-24 cores in typical processors
- **Complex Cores**: Out-of-order execution, branch prediction, large caches
- **Low Latency**: Minimize time for single thread execution
- **Control Flow**: Optimized for complex, irregular algorithms

### GPU Design (Throughput-Oriented)
- **Many Cores**: Hundreds to thousands of simple cores
- **Simple Cores**: In-order execution, minimal branch prediction
- **High Throughput**: Maximize parallel work completion
- **Data Flow**: Optimized for regular, data-parallel algorithms

## NVIDIA GPU Architecture

### Streaming Multiprocessor (SM)
The fundamental building block of NVIDIA GPUs:

**Components:**
- **CUDA Cores**: Arithmetic logic units for integer and floating-point operations
- **Special Function Units (SFUs)**: Hardware for transcendental functions
- **Load/Store Units**: Memory access operations
- **Warp Schedulers**: Thread scheduling and instruction dispatch

**Example (RTX 4090 - Ada Lovelace):**
- 128 SMs per GPU
- 128 CUDA cores per SM
- 16,384 total CUDA cores

### Memory Hierarchy

#### 1. Registers
- **Per-Thread**: Private to each thread
- **Size**: 32-bit registers, limited quantity per SM
- **Access**: Fastest memory, no latency when sufficient

#### 2. Shared Memory
- **Per-Block**: Accessible to all threads in a thread block
- **Size**: 48-163 KB per SM (architecture dependent)
- **Access**: Low latency (~1-30 cycles)
- **Use Cases**: Data sharing, reduction operations

#### 3. Global Memory
- **Device-Wide**: Accessible to all threads
- **Size**: 8-80 GB on modern GPUs
- **Access**: High latency (~200-800 cycles)
- **Bandwidth**: 500-1000 GB/s on high-end GPUs

#### 4. Constant Memory
- **Read-Only**: Cached for efficient broadcast
- **Size**: 64 KB addressable space
- **Access**: Fast when all threads read same location

#### 5. Texture Memory
- **Optimized**: For spatial locality and interpolation
- **Cached**: Texture cache optimizes 2D/3D access patterns

## Thread Hierarchy

### Thread Organization
```
Grid (Device)
├── Block 0
│   ├── Thread (0,0)
│   ├── Thread (0,1)
│   └── ...
├── Block 1
│   ├── Thread (0,0)
│   └── ...
└── ...
```

### Warp Execution
- **Warp Size**: 32 threads execute together (NVIDIA)
- **SIMT Model**: Single Instruction, Multiple Thread
- **Lock-Step**: All threads in warp execute same instruction
- **Divergence**: Branch divergence reduces efficiency

## Memory Coalescing

### Concept
Multiple memory requests from a warp are combined into fewer memory transactions.

### Coalesced Access Pattern
```
Thread 0: Address 0
Thread 1: Address 4  
Thread 2: Address 8
Thread 3: Address 12
... (sequential addresses)
```
Result: Single memory transaction

### Non-Coalesced Access Pattern
```
Thread 0: Address 100
Thread 1: Address 200
Thread 2: Address 300
... (scattered addresses)
```
Result: Multiple memory transactions

## Occupancy

### Definition
Ratio of active warps to maximum possible warps on an SM.

```
Occupancy = Active_Warps_per_SM / Max_Warps_per_SM
```

### Limiting Factors
1. **Register Usage**: More registers per thread = fewer active warps
2. **Shared Memory**: More shared memory per block = fewer blocks per SM
3. **Block Size**: Very small blocks may underutilize resources

### Optimization
- Balance resource usage to maximize occupancy
- Higher occupancy can hide memory latency
- 100% occupancy not always optimal

## GPU Architecture Generations

### NVIDIA Architectures

#### Fermi (2010)
- First architecture designed for general computing
- ECC memory support
- 16 SMs, 32 cores per SM

#### Kepler (2012)
- Dynamic parallelism
- Hyper-Q for multiple CPU cores
- Up to 15 SMs, 192 cores per SM

#### Maxwell (2014)
- Improved energy efficiency
- 16-24 SMs, 128 cores per SM
- Better branch divergence handling

#### Pascal (2016)
- 16nm FinFET process
- HBM2 memory
- NVLink interconnect
- Up to 60 SMs

#### Volta (2017)
- Tensor cores for AI workloads
- Independent thread scheduling
- 80 SMs, 64 cores per SM

#### Turing (2018)
- RT cores for ray tracing
- Variable rate shading
- 72 SMs, 64 cores per SM

#### Ampere (2020)
- 7nm process
- Third-generation Tensor cores
- Multi-instance GPU (MIG)
- 108 SMs, 64 cores per SM

#### Ada Lovelace (2022)
- 4nm process
- Third-generation RT cores
- AV1 encoding/decoding
- 128 SMs, 128 cores per SM

## Performance Characteristics

### Theoretical Peak Performance
- **Single Precision**: CUDA_cores × Base_clock × 2 ops/cycle
- **Tensor Performance**: Specialized units provide much higher throughput

### Memory Bandwidth
- **HBM3**: Up to 3.35 TB/s theoretical bandwidth
- **GDDR6X**: Up to 1 TB/s on consumer cards

### Power Consumption
- **TGP**: Total Graphics Power (150W to 600W+)
- **Performance per Watt**: Key metric for data centers

## Programming Model Implications

### SIMT Execution
- Threads in same warp execute same instruction
- Branch divergence causes serialization
- Optimize for uniform execution patterns

### Memory Access Patterns
- Coalesced global memory access is crucial
- Shared memory enables fast inter-thread communication
- Proper data layout significantly impacts performance

### Synchronization
- Warp-level primitives are efficient
- Block-level synchronization (__syncthreads())
- Grid-level synchronization requires kernel launches

## Competitive Architectures

### AMD RDNA/CDNA
- Compute Units (CUs) similar to SMs
- Wavefront size of 64 (vs. NVIDIA's 32)
- ROCm software stack

### Intel Xe
- Execution Units (EUs) 
- SYCL/DPC++ programming model
- Integrated and discrete GPUs

## Applications and Workload Characteristics

### Well-Suited Workloads
- **High Arithmetic Intensity**: More computation than memory access
- **Regular Memory Patterns**: Sequential or predictable access
- **Massive Parallelism**: Thousands of independent operations
- **SIMD-Friendly**: Same operation on different data

### Challenging Workloads
- **Irregular Memory Access**: Random memory patterns
- **Complex Control Flow**: Frequent branching
- **Fine-Grained Synchronization**: Frequent thread coordination
- **Small Problem Sizes**: Insufficient parallelism

## Future Trends

### Specialized Units
- **Tensor Cores**: Accelerate AI/ML matrix operations
- **RT Cores**: Ray-tracing acceleration
- **Video Encoders**: Hardware video compression

### Advanced Features
- **Multi-Instance GPU**: Partitioning single GPU
- **Memory Pooling**: Unified memory across devices
- **Quantum Computing**: Quantum-classical hybrid systems

## Summary

GPU architecture is fundamentally different from CPUs, optimized for:

1. **Throughput over Latency**: Many simple cores vs. few complex cores
2. **SIMT Execution**: 32-thread warps execute in lockstep
3. **Memory Hierarchy**: Multiple levels with different characteristics
4. **Coalesced Access**: Memory bandwidth optimization crucial
5. **Occupancy**: Balance resources to maximize parallel execution
6. **Evolution**: Continuous specialization for emerging workloads

Understanding GPU architecture is essential for writing efficient parallel code and achieving optimal performance.

## Next Steps

Continue to [CUDA Basics](../03_cuda_basics/README.md) to learn how to program NVIDIA GPUs using the CUDA parallel computing platform.