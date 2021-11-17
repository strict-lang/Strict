using System;
using NUnit.Framework;

namespace Strict.Compiler.Cuda.Tests;

/// <summary>
/// Kinda stupid to do parallel as the overhead is way too much to make this work.
/// Improve the problem to a more realistic benchmark actually calculating neuron weights!
/// SingleThread() * 10000000: 203ms
/// SingleThreadChunks() * 10000000: 98ms
/// ParallelCpu() * 10000000: 272ms
/// ParallelCpuChunks() * 10000000: 107ms
/// </summary>
[Category("Slow")]
public class SigmoidPerformanceTests
{
	[Test]
	public void CpuAndGpuLoops()
	{
		new TestPerformance(10000000, 100, SigmoidOutput, SigmoidGpu, _ => { }).Run();
		Console.WriteLine("output=" + output);
	}

	public void SigmoidOutput(int start, int chunkSize)
	{
		for (var n = 0; n < chunkSize; n++)
			output = 1.0f / (1.0f + (float)Math.Exp(-Input));
	}

	private float output;
	public const float Input = 0.265f;

	// ReSharper disable once MemberCanBeMadeStatic.Local
	private void SigmoidGpu(int notImplementedYet) { }
}