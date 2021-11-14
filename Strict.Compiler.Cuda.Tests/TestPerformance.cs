using System;
using System.Diagnostics;
using System.Threading.Tasks;

namespace Strict.Compiler.Cuda.Tests;

/// <summary>
/// Interestingly the chunksize doesn't matter much as long as it is 10 or more, after 100 there is almost no benefit.
/// </summary>
public record TestPerformance(int Iterations, int ChunkSize, Action<int, int> RunChunk,
	Action<int> RunGpu, Action<string> Done)
{
	public void Run()
	{
		//CheckPerformance(SingleThread);
		//CheckPerformance(SingleThreadChunks);
		CheckPerformance(ParallelCpu);
		CheckPerformance(ParallelCpuChunks);
		CheckPerformance(CudaGpu);
		CheckPerformance(CudaGpuAndCpu);
	}

	protected void CheckPerformance(Action runCode)
	{
		var watch = new Stopwatch();
		watch.Restart();
		runCode();
		watch.Stop();
		Console.WriteLine(runCode.Method.Name + " * " + Iterations + ": " +
			watch.ElapsedMilliseconds + "ms");
		Done(runCode.Method.Name);
	}

	private void SingleThread()
	{
		for (var i = 0; i < Iterations; i++)
			RunChunk(i, 1);
	}

	private void SingleThreadChunks()
	{
		for (var i = 0; i < Iterations / ChunkSize; i++)
			RunChunk(i * ChunkSize, ChunkSize);
	}

	private void ParallelCpu() => Parallel.For(0, Iterations, index => RunChunk(index, 1));

	private void ParallelCpuChunks() =>
		Parallel.For(0, Iterations / ChunkSize, index => RunChunk(index * ChunkSize, ChunkSize));

	private void CudaGpu() => RunGpu(Iterations);

	// ReSharper disable once TooManyDeclarations
	private void CudaGpuAndCpu()
	{
		//not really worth it, only after many iterations +-1%
		const float GpuFactor = 0.97f; //1f - (21f / 464f);
		var gpuIterations = (int)(Iterations * GpuFactor);
		Parallel.Invoke(() => RunGpu(gpuIterations),
			() => Parallel.For(gpuIterations / ChunkSize, Iterations / ChunkSize,
				index => RunChunk(index * ChunkSize, ChunkSize)));
	}
}