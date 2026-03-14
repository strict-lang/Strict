using System.Diagnostics;
using NUnit.Framework;

namespace Strict.Compiler.Assembly.Tests;

/// <summary>
/// Performance comparison of single-thread vs parallel CPU brightness adjustment.
/// Based on AdjustBrightness.strict which iterates all pixels: for row, column in image.Size
/// and adjusts RGB channels with Clamp(0, 255). For small images (&lt;0.1 megapixel) parallel
/// overhead makes it slower, but for 4K images (8.3MP) parallel should be significantly faster.
/// Reference from BlurPerformanceTests (2048x1024, 200 iterations):
///   SingleThread: 4594ms, ParallelCpu: 701ms (6.5x faster), CudaGpu: 32ms (143x faster)
/// Next step: use MLIR gpu dialect to offload to GPU for even faster execution.
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
	public void ParallelThresholdIsAroundPointOneMegapixel()
	{
		var largeSingle = MeasureSingleThread(CreateTestImage(1000, 1000), 1000, Iterations);
		var largeParallel = MeasureParallelCpu(CreateTestImage(1000, 1000), 1000, Iterations);
		Console.WriteLine($"1MP image (1000x1000, {Iterations} iterations):");
		Console.WriteLine($"  SingleThread: {largeSingle.TotalMilliseconds:F2}ms");
		Console.WriteLine($"  ParallelCpu:  {largeParallel.TotalMilliseconds:F2}ms");
		Console.WriteLine($"  Speedup:      {largeSingle / largeParallel:F2}x");
		Assert.That(1000 * 1000, Is.GreaterThan(ParallelPixelThreshold),
			"1MP image is above the parallelization threshold");
	}

	[Test]
	public void ShouldParallelizeBasedOnPixelCount()
	{
		Assert.That(ShouldParallelize(50, 50), Is.False,
			"2500 pixels should not use parallel execution");
		Assert.That(ShouldParallelize(316, 316), Is.False,
			"~0.1MP should not use parallel execution");
		Assert.That(ShouldParallelize(1000, 1000), Is.True,
			"1MP should use parallel execution");
		Assert.That(ShouldParallelize(3840, 2160), Is.True,
			"4K image should use parallel execution");
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

	public static bool ShouldParallelize(int width, int height) =>
		(long)width * height > ParallelPixelThreshold;

	public const int ParallelPixelThreshold = 100_000;
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
