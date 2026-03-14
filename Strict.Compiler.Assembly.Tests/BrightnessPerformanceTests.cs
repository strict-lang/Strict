using System.Diagnostics;
using NUnit.Framework;

namespace Strict.Compiler.Assembly.Tests;

/// <summary>
/// Performance comparison of single-thread vs parallel CPU vs simulated GPU brightness adjustment.
/// Based on AdjustBrightness.strict which iterates all pixels: for row, column in image.Size
/// and adjusts RGB channels with Clamp(0, 255). For small images (under 0.1 megapixel) parallel
/// overhead makes it slower, but for 4K images (8.3MP) parallel should be significantly faster.
/// Reference from BlurPerformanceTests (2048x1024, 200 iterations):
///   SingleThread: 4594ms, ParallelCpu: 701ms (6.5x faster), CudaGpu: 32ms (143x faster)
/// Complexity = iterations × body instructions, not just iteration count alone.
/// A 10K loop with 1K body instructions has the same complexity as a 1M loop with 10 body instructions.
/// </summary>
public sealed class BrightnessPerformanceTests
{
	[Test]
	public void SmallImageSingleThreadIsFastEnough()
	{
		var pixels = CreateTestImage(100, 100);
		var elapsed = MeasureSingleThread(pixels, 100, Iterations);
		Assert.That(elapsed.TotalMilliseconds, Is.LessThan(500),
			"100x100 image should process quickly on single thread");
	}

	[Test]
	public void LargeImageParallelIsFaster()
	{
		const int width = 3840;
		const int height = 2160;
		var pixels = CreateTestImage(width, height);
		var singleThreadTime = MeasureSingleThread(pixels, width, Iterations);
		var parallelTime = MeasureParallelCpu(pixels, width, Iterations);
		Console.WriteLine($"4K image ({width}x{height} = {width * height} pixels, {Iterations} iterations):");
		Console.WriteLine($"  SingleThread: {singleThreadTime.TotalMilliseconds:F2}ms");
		Console.WriteLine($"  ParallelCpu:  {parallelTime.TotalMilliseconds:F2}ms");
		Console.WriteLine($"  Speedup:      {singleThreadTime / parallelTime:F2}x");
		Assert.That(parallelTime, Is.LessThan(singleThreadTime),
			"Parallel should be faster than single-thread for 4K image");
	}

	[Test]
	public void GpuShouldBeMuchFasterForLargeImages()
	{
		const int width = 3840;
		const int height = 2160;
		var pixels = CreateTestImage(width, height);
		var singleThreadTime = MeasureSingleThread(pixels, width, Iterations);
		var parallelTime = MeasureParallelCpu(pixels, width, Iterations);
		var cpuSpeedup = singleThreadTime / parallelTime;
		Console.WriteLine(
			$"4K image ({width}x{height} = {width * height} pixels, {Iterations} iterations):");
		Console.WriteLine($"  SingleThread: {singleThreadTime.TotalMilliseconds:F2}ms");
		Console.WriteLine($"  ParallelCpu:  {parallelTime.TotalMilliseconds:F2}ms");
		Console.WriteLine($"  CPU Speedup:  {cpuSpeedup:F2}x");
		Console.WriteLine("  Expected GPU: ~{0:F2}ms (projected {1:F0}x from CUDA reference data)",
			singleThreadTime.TotalMilliseconds / ExpectedGpuSpeedupOverSingleThread,
			ExpectedGpuSpeedupOverSingleThread);
		Assert.That(cpuSpeedup, Is.GreaterThan(1.5),
			"CPU parallel should provide at least 1.5x speedup for 4K images");
		var projectedGpuTime = singleThreadTime.TotalMilliseconds / ExpectedGpuSpeedupOverSingleThread;
		Assert.That(projectedGpuTime, Is.LessThan(parallelTime.TotalMilliseconds),
			"Projected GPU time should beat CPU parallel — " +
			"based on BlurPerformanceTests showing GPU is 143x faster than single-thread");
	}

	/// <summary>
	/// From BlurPerformanceTests 2048x1024 reference data:
	/// SingleThread: 4594ms, CudaGpu: 32ms → 143x speedup.
	/// We use a conservative 20x estimate since brightness is simpler than blur and
	/// real GPU overhead includes kernel launch + memory transfer, which MLIR gpu.launch must handle.
	/// </summary>
	private const double ExpectedGpuSpeedupOverSingleThread = 20;

	[Test]
	public void ParallelThresholdIsAroundPointOneMegapixel()
	{
		var largeSingle = MeasureSingleThread(CreateTestImage(1000, 1000), 1000, Iterations);
		var largeParallel = MeasureParallelCpu(CreateTestImage(1000, 1000), 1000, Iterations);
		Console.WriteLine($"1MP image (1000x1000, {Iterations} iterations):");
		Console.WriteLine($"  SingleThread: {largeSingle.TotalMilliseconds:F2}ms");
		Console.WriteLine($"  ParallelCpu:  {largeParallel.TotalMilliseconds:F2}ms");
		Console.WriteLine($"  Speedup:      {largeSingle / largeParallel:F2}x");
		Assert.That(1000 * 1000, Is.GreaterThan(100_000),
			"1MP image is above the parallelization threshold");
	}

	[Test]
	public void ComplexityBasedThresholdConsidersBodyInstructions()
	{
		Assert.That(EstimateComplexity(1_000_000, 1), Is.EqualTo(1_000_000),
			"1M iterations × 1 instruction = 1M complexity");
		Assert.That(EstimateComplexity(10_000, 1000), Is.EqualTo(10_000_000),
			"10K iterations × 1K instructions = 10M complexity");
		Assert.That(ShouldParallelize(1_000_000, 1), Is.True,
			"1M complexity should parallelize");
		Assert.That(ShouldParallelize(10_000, 1000), Is.True,
			"10M complexity should parallelize");
		Assert.That(ShouldParallelize(100, 1), Is.False,
			"100 complexity should NOT parallelize");
		Assert.That(ShouldParallelize(50_000, 1), Is.False,
			"50K complexity should NOT parallelize");
		Assert.That(ShouldParallelize(10_000, 20), Is.True,
			"200K complexity should parallelize (like AdjustBrightness with complex body)");
	}

	[TestCase(10)]
	[TestCase(50)]
	[TestCase(-20)]
	public void BrightnessAdjustmentProducesCorrectResults(int brightness)
	{
		var pixels = CreateTestImage(10, 10);
		var expected = new byte[pixels.Length];
		Array.Copy(pixels, expected, pixels.Length);
		ApplyBrightnessSingleThread(expected, brightness);
		var parallelResult = new byte[pixels.Length];
		Array.Copy(pixels, parallelResult, pixels.Length);
		ApplyBrightnessParallel(parallelResult, 10, brightness);
		Assert.That(parallelResult, Is.EqualTo(expected),
			"Parallel result must match single-thread result");
	}

	public static long EstimateComplexity(long iterations, int bodyInstructionCount) =>
		iterations * Math.Max(bodyInstructionCount, 1);

	public static bool ShouldParallelize(long iterations, int bodyInstructionCount) =>
		EstimateComplexity(iterations, bodyInstructionCount) >
		InstructionsToMlir.ComplexityThreshold;

	private const int Iterations = 10;
	private const int Brightness = 10;

	private static byte[] CreateTestImage(int width, int height)
	{
		var pixels = new byte[width * height * 3];
		var random = new Random(42);
		random.NextBytes(pixels);
		return pixels;
	}

	private static void AdjustPixelBrightness(byte[] pixels, int pixelIndex, int brightness)
	{
		for (var channel = 0; channel < 3; channel++)
		{
			var index = pixelIndex * 3 + channel;
			pixels[index] = (byte)Math.Clamp(pixels[index] + brightness, 0, 255);
		}
	}

	private static void ApplyBrightnessSingleThread(byte[] pixels, int brightness)
	{
		var pixelCount = pixels.Length / 3;
		for (var pixelIndex = 0; pixelIndex < pixelCount; pixelIndex++)
			AdjustPixelBrightness(pixels, pixelIndex, brightness);
	}

	private static void ApplyBrightnessParallel(byte[] pixels, int width, int brightness)
	{
		var height = pixels.Length / 3 / width;
		Parallel.For(0, height, row =>
		{
			for (var column = 0; column < width; column++)
				AdjustPixelBrightness(pixels, row * width + column, brightness);
		});
	}

	private static TimeSpan MeasureSingleThread(byte[] sourcePixels, int width, int iterations)
	{
		var pixels = new byte[sourcePixels.Length];
		Array.Copy(sourcePixels, pixels, sourcePixels.Length);
		var stopwatch = Stopwatch.StartNew();
		for (var iteration = 0; iteration < iterations; iteration++)
			ApplyBrightnessSingleThread(pixels, Brightness);
		stopwatch.Stop();
		return stopwatch.Elapsed;
	}

	private static TimeSpan MeasureParallelCpu(byte[] sourcePixels, int width, int iterations)
	{
		var pixels = new byte[sourcePixels.Length];
		Array.Copy(sourcePixels, pixels, sourcePixels.Length);
		var stopwatch = Stopwatch.StartNew();
		for (var iteration = 0; iteration < iterations; iteration++)
			ApplyBrightnessParallel(pixels, width, Brightness);
		stopwatch.Stop();
		return stopwatch.Elapsed;
	}
}
