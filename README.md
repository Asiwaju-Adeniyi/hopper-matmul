# Hopper MatMul

Since AI labs still very much possess a high number of Hopper GPUs, I thought I'd do a from-scratch exploration of high-performance matrix multiplication on NVIDIA Hopper GPUs, specifically the **H100**.

In this repository, I follow the ideas and optimization progression from Pranjal's [Outperforming cuBLAS on H100](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) worklog, while implementing the kernels independently to build a deeper understanding of Hopper GPU architecture and CUDA performance optimization.

I want to build a sequence of increasingly optimized CUDA matrix multiplication kernels and understand **why** each optimization improves performance.

The project will explore Hopper-specific features such as:

- Thread Block Clusters
- Tensor Memory Accelerator (TMA)
- Tensor Cores
- Warp-Group Matrix Multiply-Accumulate (WGMMA)
- Warp specialization
- Asynchronous data movement
- Shared-memory pipelining
- Register tiling
- Persistent kernels
- Double/triple buffering
- Memory and compute latency hiding

Each optimization will be benchmarked against previous implementations and, where useful, against cuBLAS.

## Matrix Multiplication

The core operation is:

**C = A × B**

For matrices:

- `A`: `M × K`
- `B`: `K × N`
- `C`: `M × N`

The initial implementations will use FP32 before progressively exploring lower-precision inputs and Hopper Tensor Core execution.

## Learning Approach

Rather than jumping directly into an optimized Hopper kernel, the project will be developed incrementally.

### Phase 1 — Naive CUDA MatMul

Implement a basic CUDA matrix multiplication kernel.

Focus:

- CUDA thread hierarchy
- Global memory accesses
- Thread-to-output mapping
- Correctness
- Basic benchmarking

### Phase 2 — Shared-Memory Tiling

Introduce block tiling using shared memory.

Focus:

- Global → shared memory movement
- Shared-memory reuse
- Memory coalescing
- Synchronization
- Arithmetic intensity

### Phase 3 — Register Tiling

Move parts of the computation into registers.

Focus:

- Register reuse
- Per-thread tiles
- Register pressure
- Occupancy

### Phase 4 — Asynchronous Pipelines

Introduce asynchronous data movement and multi-stage buffering.

Focus:

- Overlapping memory movement with computation
- Pipeline stages
- `cp.async`
- Synchronization mechanisms

### Phase 5 — Hopper Memory Movement

Move toward Hopper-specific memory operations.

Focus:

- Tensor Memory Accelerator (TMA)
- Tensor memory access patterns
- Shared-memory layouts
- Asynchronous tensor transfers

### Phase 6 — WGMMA

Introduce Hopper's Warp-Group Matrix Multiply-Accumulate instructions.

Focus:

- Warp groups
- WGMMA
- Tensor Core execution
- Accumulator layouts
- Register requirements

### Phase 7 — Warp-Specialized MatMul

Separate the responsibilities of different warps/warp groups.

For example:

- Producer warps → move data
- Consumer warps → perform matrix multiplication

Focus:

- Warp specialization
- Producer/consumer pipelines
- Synchronization
- Latency hiding

### Phase 8 — Persistent Kernel

Explore persistent execution and keeping work resident on the GPU.

Focus:

- Persistent thread blocks
- Work distribution
- Grid synchronization
- Improving GPU utilization

### Phase 9 — Final Optimized Kernel

Combine the most effective techniques into a relatively compact Hopper GEMM kernel.

The final implementation will be compared against:

- Previous kernels in this repository
- cuBLAS
- Performance reported in the reference worklog

## Benchmarking

The primary benchmark will initially use:

`N = 4096`

Additional matrix sizes will be tested to understand how the kernels behave under different workloads.

Metrics will include:

- Execution time
- TFLOP/s
- Speedup relative to previous kernels
- Percentage of cuBLAS performance

Results will be recorded as the project progresses.

## Repository Structure

hopper-matmul/
├── README.md
├── CMakeLists.txt
├── kernels/
│   ├── 01_naive.cu
│   ├── 02_shared_memory.cu
│   ├── 03_register_tiling.cu
│   ├── 04_async_pipeline.cu
│   ├── 05_tma.cu
│   ├── 06_wgmma.cu
│   ├── 07_warp_specialized.cu
│   ├── 08_persistent.cu
│   └── 09_final.cu
├── benchmarks/
│   └── benchmark.cu
├── results/
│   └── results.csv
└── notes/
    ├── hopper_architecture.md
    ├── tma.md
    ├── wgmma.md
    ├── warp_specialization.md
    └── optimization_log.md

## Hardware

The target architecture is NVIDIA Hopper:

- GPU: NVIDIA H100
- Compute Capability: `sm_90`

The kernels are therefore intended to be compiled specifically for Hopper where required.

## Reference

This project is inspired by:

**Pranjal — Outperforming cuBLAS on H100: A Worklog**

[https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)

