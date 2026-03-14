# GPU Support Plan for Strict via MLIR

## Overview

Strict's MLIR backend compiles bytecode to MLIR's arith/func/scf dialects, which are then lowered
through LLVM to native executables. This plan extends the pipeline to support GPU offloading using
MLIR's GPU dialect ecosystem, enabling automatic parallelization of compute-intensive loops like
image processing operations.

## Current State (Completed)

### Phase 0: CPU Baseline and Parallel Foundation
- **Three compiler backends**: MLIR (default), LLVM IR, and NASM — all generating 2-4KB executables
- **scf.for loops**: Range-based loops in Strict (`for row, column in image.Size`) now compile to
  `scf.for` in MLIR
- **scf.parallel auto-detection**: Loops with >100K iterations automatically emit `scf.parallel`
  instead of `scf.for`, enabling the MLIR pass pipeline to optimize for multiple CPU cores
- **Performance validated**: BrightnessPerformanceTests confirms 2.5x+ speedup for 4K images
  (3840x2160) using parallel CPU execution vs single-thread
- **Lowering pipeline**: mlir-opt includes `--convert-scf-to-cf` to lower SCF constructs to
  control flow before LLVM lowering

### Reference Performance (from BlurPerformanceTests, 2048x1024 image, 200 iterations)
```
SingleThread:      4594ms (baseline)
ParallelCpu:        701ms (6.5x faster)
CudaGpu:             32ms (143x faster)
CudaGpuAndCpu:       29ms (158x faster)
```

## Phase 1: CPU Parallel via OpenMP (Next)

**Goal**: Use MLIR's OpenMP lowering to generate multi-threaded native code from `scf.parallel`.

### Implementation
1. Add `--convert-scf-to-openmp` pass to MlirLinker before `--convert-arith-to-llvm`
2. Add `--convert-openmp-to-llvm` pass after OpenMP conversion
3. Link with OpenMP runtime (`-fopenmp` flag to clang)
4. Benchmark against BrightnessPerformanceTests to confirm native parallel speedup

### MLIR Pipeline Change
```
Current:  scf.parallel → scf-to-cf → cf-to-llvm → LLVM IR → clang
Phase 1:  scf.parallel → scf-to-openmp → openmp-to-llvm → LLVM IR → clang -fopenmp
```

### Complexity Threshold
- Loops below 100K iterations: remain as `scf.for` (sequential)
- Loops above 100K iterations: `scf.parallel` → OpenMP threads
- This threshold matches the ~0.1 megapixel boundary where parallelization overhead breaks even

## Phase 2: GPU Offloading via MLIR GPU Dialect

**Goal**: Offload parallelizable loops to GPU when available and beneficial.

### Implementation
1. Add `gpu.launch` / `gpu.launch_func` emission for loops exceeding a GPU-worthy threshold
   (e.g., >1M iterations or >1MP image)
2. Emit `gpu.alloc` / `gpu.memcpy` for data transfer between host and device
3. Generate GPU kernel functions from loop bodies
4. Add MLIR lowering passes:
   - `--gpu-kernel-outlining` (extract loop body into GPU kernel)
   - `--convert-gpu-to-nvvm` (for NVIDIA GPUs via NVVM/PTX)
   - `--gpu-to-llvm` (GPU runtime calls)
   - `--convert-nvvm-to-llvm` (NVVM intrinsics to LLVM)

### MLIR Pipeline for GPU
```
scf.parallel → gpu-map-parallel-loops → gpu-kernel-outlining
             → convert-gpu-to-nvvm → gpu-to-llvm → LLVM IR
             → clang -lcuda (or -L/path/to/cuda)
```

### Data Transfer Optimization
- Minimize host↔device copies by analyzing data flow
- For image processing: copy image to GPU once, run kernel, copy result back
- Use `gpu.alloc` + `gpu.memcpy` for explicit memory management
- Future: use unified memory (`gpu.alloc host_shared`) when available

## Phase 3: Automatic Optimization Path Selection

**Goal**: The Strict compiler automatically decides the fastest execution strategy per loop.

### Complexity Analysis
The compiler should track instruction complexity to decide optimization path:

| Complexity Level | Pixel Count | Strategy | Example |
|-----------------|-------------|----------|---------|
| Trivial | < 10K | Sequential CPU | Thumbnail processing |
| Low | 10K - 100K | Sequential CPU | Small images |
| Medium | 100K - 1M | Parallel CPU (OpenMP) | HD images |
| High | > 1M | GPU offload | 4K/8K images |

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
- Verify `gpu.launch` emission for GPU-targeted loops
- Verify correct lowering pass pipeline configuration ✓

### Performance Tests (BrightnessPerformanceTests)
- Single-thread vs Parallel CPU for various image sizes ✓
- CPU vs GPU execution comparison
- Threshold detection validation ✓
- Correctness: parallel and GPU results match sequential ✓

### Integration Tests
- End-to-end: AdjustBrightness.strict → MLIR → native executable → correct output
- Binary size validation: GPU-enabled executables remain reasonable
- Platform-specific: test on Linux/Windows/macOS with and without GPU

## Dependencies

| Phase | MLIR Passes Required | Runtime Libraries |
|-------|---------------------|-------------------|
| 1 | scf-to-openmp, openmp-to-llvm | OpenMP runtime (libomp) |
| 2 | gpu-kernel-outlining, convert-gpu-to-nvvm, gpu-to-llvm | CUDA runtime |
| 3 | (analysis passes, no new lowering) | Same as Phase 2 |
| 4 | convert-gpu-to-rocdl, convert-gpu-to-spirv | ROCm, Level Zero |
