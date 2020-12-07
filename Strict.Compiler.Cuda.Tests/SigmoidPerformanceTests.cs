using System;
using System.Diagnostics;
using System.Threading.Tasks;
using NUnit.Framework;

namespace Strict.Compiler.Cuda.Tests
{
	/// <summary>
	/// Kinda stupid to do parallel as the overhead is way too much to make this work.
	/// Improve the problem to a more realistic benchmark actually calculating neuron weights!
	/// </summary>
	[Category("Slow")]
	public class SigmoidPerformanceTests
	{
		[Test]
		public void CpuAndGpuLoops()
		{
			CheckPerformance(SingleThread);
			CheckPerformance(ParallelCpu);
			CheckPerformance(ParallelCpuChunks);
			CheckPerformance(CudaGpu);
			CheckPerformance(CudaGpuAndCpu);
		}

		private void CheckPerformance(Action runCode)
		{
			var watch = new Stopwatch();
			watch.Restart();
			runCode();
			watch.Stop();
			Console.WriteLine("Sigmoid " + NumberOfIterations + " calls on " + runCode.Method + ": " +
				watch.ElapsedMilliseconds + "ms, output=" + output);
		}
		
		public const int NumberOfIterations = 100000000;
		public readonly Sigmoid sigmoid = new();
		public class Sigmoid
		{
			public float Output(float x) => 1.0f / (1.0f + (float)Math.Exp(-x));
		}

		private void SingleThread()
		{
			for (var i = 0; i < NumberOfIterations; i++)
				output = sigmoid.Output(Input);
		}

		public float output = 0f;
		public const float Input = 0.265f;

		private void ParallelCpu() => Parallel.For(0, NumberOfIterations, index => output = sigmoid.Output(Input));
		
		private void ParallelCpuChunks() => Parallel.For(0, NumberOfIterations/ChunkSize, index =>
		{
			for (var i = 0; i < ChunkSize; i++)
				output = sigmoid.Output(Input);
		});
		/// <summary>
		/// Interestingly the chunksize doesn't matter much as long as it is 10 or more, after 100 there is almost no benefit.
		/// </summary>
		public const int ChunkSize = 100;
		
		private void CudaGpu() { }
		private void CudaGpuAndCpu() { }
	}
}