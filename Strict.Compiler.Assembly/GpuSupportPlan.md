# GPU Support Plan for Strict via MLIR

## Overview

Strict's MLIR backend compiles bytecode to MLIR's arith/func/scf dialects, which are then lowered
through LLVM to native executables. This plan extends the pipeline to support GPU offloading using
MLIR's GPU dialect ecosystem, enabling automatic parallelization of compute-intensive loops like
image processing operations.

## Current State (Completed)

### Phase 0: CPU Baseline and Parallel Foundation
- **Three compiler backends**: MLIR (default), LLVM IR, and NASM — all generating 2-4KB executables
- **scf.for loops**: Range-based loops in Strict (`for row, column in image.Size`) compile to `scf.for`
- **scf.parallel auto-detection**: Loops emit `scf.parallel` when complexity (iterations × body
  instruction count) exceeds 100K threshold
- **Complexity-based threshold**: A 10K loop with 20 body instructions (200K) parallelizes, while
  50K with 1 instruction (50K) stays sequential
- **Lowering pipeline**: mlir-opt includes `--convert-scf-to-cf` to lower SCF to control flow

### Phase 1: CPU Parallel via OpenMP
- `scf.parallel` emitted for 100K–10M complexity range
- OpenMP lowering ready: `--convert-scf-to-openmp` → `--convert-openmp-to-llvm` → clang -fopenmp
- Complexity threshold exposed as `InstructionsToMlir.ComplexityThreshold = 100_000`

### Phase 2: GPU Offloading via MLIR GPU Dialect (In Progress)
- **3-tier loop strategy implemented** in InstructionsToMlir:
  - Complexity < 100K → `scf.for` (sequential)
  - 100K ≤ Complexity < 10M → `scf.parallel` (CPU parallel)
  - Complexity ≥ 10M → `gpu.launch` (GPU offload)
- **gpu.launch emission**: Generates grid/block dimensions from iteration count
  - Block size: 256 threads, grid: ⌈iterations/256⌉ blocks
  - Uses `gpu.block_id x`, `gpu.thread_id x` to compute global thread index
  - Body terminates with `gpu.terminator`
- **GPU lowering passes** in MlirLinker.BuildMlirOptArgsWithGpu:
  - `--gpu-kernel-outlining` → `--convert-gpu-to-nvvm` → `--gpu-to-llvm` → `--convert-nvvm-to-llvm`
- **GPU threshold**: `InstructionsToMlir.GpuComplexityThreshold = 10_000_000`
- **Example**: AdjustBrightness.strict Run method processes 1280×720 (921,600 pixels) — with
  20+ body instructions per pixel this exceeds the 10M GPU threshold

### Reference Performance (from BlurPerformanceTests, 2048x1024 image, 200 iterations)
```
SingleThread:      4594ms (baseline)
ParallelCpu:        701ms (6.5x faster)
CudaGpu:             32ms (143x faster)
CudaGpuAndCpu:       29ms (158x faster)
```

## Remaining Work

### GPU Memory Management
- Emit `gpu.alloc` / `gpu.memcpy` for data transfer between host and device
- For image processing: copy image to GPU once, run kernel, copy result back
- Future: use unified memory (`gpu.alloc host_shared`) when available

### Runner Integration
- Wire `BuildMlirOptArgsWithGpu` when MLIR contains `gpu.launch` ops
- Add `-gpu` CLI flag to enable GPU compilation path
- Link with CUDA runtime (`-lcuda` or `-L/path/to/cuda` in clang args)
- Fall back to CPU parallel when no GPU toolchain is available

### MLIR Pipeline for GPU (end-to-end)
```
gpu.launch → gpu-kernel-outlining → convert-gpu-to-nvvm → gpu-to-llvm
           → convert-nvvm-to-llvm → LLVM IR → clang -lcuda
```

#### Phase 3: Automatic Optimization Path Selection

**Goal**: The Strict compiler automatically decides the fastest execution strategy per loop.

### Complexity Analysis (Implemented)
The compiler tracks `iterations × body instruction count` as total complexity to decide:

| Total Complexity | Strategy | Constant | Example |
|-----------------|----------|----------|---------|
| < 100K | scf.for (sequential) | ComplexityThreshold | Small images, simple loops |
| 100K - 10M | scf.parallel (CPU) | ComplexityThreshold | HD images, moderate body |
| > 10M | gpu.launch (GPU) | GpuComplexityThreshold | 4K images, complex per-pixel |

Key insight: A 10K iteration loop with 1K body instructions (10M complexity) is a better
GPU candidate than a 1M iteration loop with 1 instruction (1M complexity), because the
GPU kernel launch overhead is amortized across more per-thread work.

### Parallelizability Detection
Not all loops can be parallelized. The compiler must verify:
- **No side effects**: No file I/O, logging, or system calls in loop body
- **No data dependencies**: Each iteration reads/writes independent data
- **No mutable shared state**: Only thread-local mutations allowed
- **Functional loop body**: Pure computation on input → output mapping

### Cases that MUST remain sequential
- Loops with `log()` / `System.Write` calls (output ordering matters)
- Loops that modify shared mutable variables (accumulation patterns)
- Loops with file system access
- Loops with network calls or external system interaction
- Iterators with ordering dependencies (e.g., linked list traversal)

## Phase 4: Multi-Backend GPU Support

**Goal**: Support multiple GPU vendors and compute targets beyond NVIDIA CUDA.

### Targets
- **NVIDIA**: `convert-gpu-to-nvvm` → PTX → CUDA runtime
- **AMD**: `convert-gpu-to-rocdl` → ROCm/HIP runtime
- **Intel**: `convert-gpu-to-spirv` → SPIR-V → Level Zero/OpenCL
- **Apple**: `convert-gpu-to-metal` (when MLIR Metal backend matures)
- **TPU**: `convert-to-tpu` (Google TPU via XLA/StableHLO)

### Runtime Detection
```
1. Check for GPU availability at compile time or startup
2. Query GPU memory and compute capability
3. Fall back to CPU parallel if no suitable GPU found
4. Generate both CPU and GPU code paths (like CudaGpuAndCpu approach)
```

## Phase 5: Advanced Optimizations

### Kernel Fusion
- Merge adjacent parallelizable loops into single GPU kernel
- Example: AdjustBrightness + AdjustContrast in one pass over pixel data

### Memory Layout Optimization
- Struct-of-Arrays (SoA) for GPU-friendly memory access
- Image data as separate R, G, B channels rather than interleaved RGB
- Automatic SoA↔AoS conversion at host↔device boundary

### Hybrid CPU+GPU Execution
- Split work between CPU and GPU (as in BlurPerformanceTests' CudaGpuAndCpu)
- Auto-tune split ratio based on relative device speeds
- Pipeline data transfer with computation overlap

## Testing Strategy

### Unit Tests (InstructionsToMlirTests)
- Verify `scf.for` emission for sequential loops ✓
- Verify `scf.parallel` emission for large loops ✓
- Verify `gpu.launch` emission for GPU-targeted loops ✓
- Verify `gpu.terminator` and block/grid dimensions ✓
- Verify medium complexity stays scf.parallel not gpu.launch ✓
- Verify correct lowering pass pipeline configuration ✓
- Verify GPU-specific lowering passes (gpu-kernel-outlining, convert-gpu-to-nvvm) ✓

### Performance Tests (BlurPerformanceTests)
- Single-thread vs Parallel CPU blur for various image sizes ✓
- Complexity-based parallelization threshold ✓
- Correctness: parallel results match sequential ✓

### Integration Tests (Next)
- End-to-end: AdjustBrightness.strict → MLIR → native GPU executable → correct output
- Binary size validation: GPU-enabled executables remain reasonable
- Platform-specific: test on Linux/Windows/macOS with and without GPU

## Dependencies

| Phase | MLIR Passes Required | Runtime Libraries |
|-------|---------------------|-------------------|
| 1 | scf-to-openmp, openmp-to-llvm | OpenMP runtime (libomp) |
| 2 | gpu-kernel-outlining, convert-gpu-to-nvvm, gpu-to-llvm | CUDA runtime |
| 3 | (analysis passes, no new lowering) | Same as Phase 2 |
| 4 | convert-gpu-to-rocdl, convert-gpu-to-spirv | ROCm, Level Zero |
