# GPU Support Plan for Strict via MLIR

## Overview

Strict's MLIR backend compiles bytecode to MLIR's arith/func/scf/gpu dialects, which are then lowered
through LLVM to native executables. The pipeline supports automatic GPU offloading of compute-intensive
loops like image processing operations, using MLIR's GPU dialect for kernel generation and device
memory management.

## Implemented

### Phase 0: CPU Baseline and Parallel Foundation
- **Three compiler backends**: MLIR (default), LLVM IR, and NASM — all generating 2-4KB executables
- **scf.for loops**: Range-based loops in Strict compile to `scf.for` in MLIR
- **scf.parallel auto-detection**: Loops emit `scf.parallel` when complexity (iterations × body
  instruction count) exceeds 100K threshold
- **Complexity-based threshold**: `InstructionsToMlir.ComplexityThreshold = 100_000`
- **Lowering pipeline**: mlir-opt includes `--convert-scf-to-cf` to lower SCF to control flow

### Phase 1: CPU Parallel via OpenMP
- `scf.parallel` emitted for 100K–10M complexity range
- OpenMP lowering path: `--convert-scf-to-openmp` → `--convert-openmp-to-llvm` → clang -fopenmp

### Phase 2: GPU Offloading via MLIR GPU Dialect
- **3-tier loop strategy** in InstructionsToMlir:
  - Complexity < 100K → `scf.for` (sequential)
  - 100K ≤ Complexity < 10M → `scf.parallel` (CPU parallel)
  - Complexity ≥ 10M → `gpu.launch` (GPU offload)
- **Correct gpu.launch emission** using MLIR gpu.launch region arguments:
  - `blocks(%bx, %by, %bz) in (%grid_x = ..., ...)` / `threads(%tx, %ty, %tz) in (%block_x = ..., ...)`
  - Global thread ID: `arith.muli %bx, %block_x` then `arith.addi`, **not** `gpu.block_id`/`gpu.thread_id`
  - Bounds checking: `arith.cmpi ult` + `scf.if` guards kernel body
  - Body terminates with `gpu.terminator`
- **GPU memory management pipeline**:
  - `memref.alloc` → host buffer allocation
  - `gpu.alloc` → device buffer allocation
  - `gpu.memcpy` host→device (before kernel) and device→host (after kernel)
  - `gpu.dealloc` + `memref.dealloc` → cleanup
- **Module attribute**: `module attributes {gpu.container_module}` emitted when GPU ops present,
  required for `--gpu-kernel-outlining` pass
- **MlirLinker auto-detection**: `CreateExecutable` reads MLIR content, detects `gpu.launch`,
  automatically selects GPU pass pipeline and CUDA linking
- **GPU lowering passes** (in order):
  ```
  --gpu-kernel-outlining → --convert-scf-to-cf → --convert-gpu-to-nvvm
  → --gpu-to-llvm → --convert-nvvm-to-llvm → --convert-memref-to-llvm
  → --convert-arith-to-llvm → --convert-func-to-llvm → --convert-cf-to-llvm
  ```
- **GPU linking**: clang with `-lcuda -lcudart` for NVIDIA targets
- **GPU threshold**: `InstructionsToMlir.GpuComplexityThreshold = 10_000_000`
- **Example**: AdjustBrightness.strict Run method processes 1280×720 (921,600 pixels)

### Reference Performance (from Strict.Compiler.Cuda.Tests, 2048x1024 image, 200 iterations)
```
SingleThread:      4594ms (baseline)
ParallelCpu:        701ms (6.5x faster)
CudaGpu:             32ms (143x faster)
CudaGpuAndCpu:       29ms (158x faster)
```

## Remaining Work

### Array/Memref Integration with Bytecode
- Current bytecode is scalar-only (f64 registers); no array/buffer instructions
- For true data-parallel GPU execution, bytecode needs `memref.load`/`memref.store` per element
- Currently the memref pipeline is structural boilerplate around scalar loop body
- Next: Add array-typed bytecode instructions so each GPU thread processes a different data element

### Phase 3: Automatic Optimization Path Selection

**Goal**: The Strict compiler automatically decides the fastest execution strategy per loop.

#### Complexity Analysis (Implemented)
| Total Complexity | Strategy | Constant | Example |
|-----------------|----------|----------|---------|
| < 100K | scf.for (sequential) | ComplexityThreshold | Small images, simple loops |
| 100K - 10M | scf.parallel (CPU) | ComplexityThreshold | HD images, moderate body |
| > 10M | gpu.launch (GPU) | GpuComplexityThreshold | 4K images, complex per-pixel |

#### Parallelizability Detection
Not all loops can be parallelized. The compiler must verify:
- **No side effects**: No file I/O, logging, or system calls in loop body
- **No data dependencies**: Each iteration reads/writes independent data
- **No mutable shared state**: Only thread-local mutations allowed
- **Functional loop body**: Pure computation on input → output mapping

### Phase 4: Multi-Backend GPU Support

**Goal**: Support multiple GPU vendors and compute targets beyond NVIDIA CUDA.

| Target | MLIR Pass | Runtime |
|--------|-----------|---------|
| NVIDIA | `convert-gpu-to-nvvm` → PTX | CUDA runtime |
| AMD | `convert-gpu-to-rocdl` | ROCm/HIP |
| Intel | `convert-gpu-to-spirv` | Level Zero/OpenCL |

### Phase 5: Advanced Optimizations
- **Kernel Fusion**: Merge adjacent parallelizable loops into single GPU kernel
- **Memory Layout**: SoA for GPU-friendly access, auto SoA↔AoS conversion
- **Hybrid CPU+GPU**: Split work between CPU and GPU, auto-tune split ratio

## Testing Strategy

### Unit Tests (InstructionsToMlirTests) — All Passing ✓
- `scf.for` emission for sequential loops
- `scf.parallel` emission for large loops
- `gpu.launch` emission with correct region arguments
- Global thread ID computation: `blockIdx * blockDim + threadIdx`
- Bounds checking: `arith.cmpi ult` + `scf.if`
- Memory management: `memref.alloc`, `gpu.alloc`, `gpu.memcpy`, `gpu.dealloc`, `memref.dealloc`
- `gpu.container_module` attribute on module
- Medium complexity stays `scf.parallel` not `gpu.launch`
- GPU lowering passes including `--convert-memref-to-llvm`
- Non-GPU modules use plain `module {}`

### Performance Tests (BlurPerformanceTests)
- Single-thread vs Parallel CPU blur for various image sizes ✓
- Correctness: parallel results match sequential ✓

## Dependencies

| Phase | MLIR Passes Required | Runtime Libraries |
|-------|---------------------|-------------------|
| 1 | scf-to-openmp, openmp-to-llvm | OpenMP runtime (libomp) |
| 2 | gpu-kernel-outlining, convert-gpu-to-nvvm, gpu-to-llvm, convert-memref-to-llvm | CUDA runtime |
| 3 | (analysis passes, no new lowering) | Same as Phase 2 |
| 4 | convert-gpu-to-rocdl, convert-gpu-to-spirv | ROCm, Level Zero |
