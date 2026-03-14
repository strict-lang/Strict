using System.Diagnostics;
using NUnit.Framework;

namespace Strict.Compiler.Assembly.Tests;

/// <summary>
/// 5x5 box blur performance test comparing single-thread vs parallel CPU vs projected GPU.
/// Blur has a much heavier per-pixel body than brightness (~30 instructions: 25 reads + sum + divide)
/// making it the perfect case for complexity-based parallelization. Even a 200x200 image (40K pixels)
/// reaches 1.2M complexity (40K × 30), well above the 100K threshold.
/// Reference from Strict.Compiler.Cuda.Tests/BlurPerformanceTests (2048x1024, 200 iterations):
///   SingleThread: 4594ms, ParallelCpu: 701ms (6.5x), CudaGpu: 32ms (143x)
/// </summary>
public sealed class BlurPerformanceTests
{
	[Test]
	public void SmallImageBlurIsFastEnough()
	{
		var pixels = CreateTestImage(100, 100);
		var elapsed = MeasureSingleThread(pixels, 100, 100, Iterations);
		Assert.That(elapsed.TotalMilliseconds, Is.LessThan(500),
			"100x100 blur should complete quickly on single thread");
	}

	[Test]
	public void LargeImageBlurParallelIsFaster()
	{
		const int width = 2048;
		const int height = 1024;
		var pixels = CreateTestImage(width, height);
		var singleTime = MeasureSingleThread(pixels, width, height, Iterations);
		var parallelTime = MeasureParallelCpu(pixels, width, height, Iterations);
		var speedup = singleTime / parallelTime;
		Console.WriteLine($"2K image blur ({width}x{height} = {width * height} pixels, {Iterations} iterations):");
		Console.WriteLine($"  SingleThread: {singleTime.TotalMilliseconds:F2}ms");
		Console.WriteLine($"  ParallelCpu:  {parallelTime.TotalMilliseconds:F2}ms");
		Console.WriteLine($"  Speedup:      {speedup:F2}x");
		Assert.That(parallelTime, Is.LessThan(singleTime),
			"Parallel should be faster than single-thread for 2K image blur");
	}

	[Test]
	public void FourKBlurShowsStrongParallelAndGpuProjection()
	{
		const int width = 3840;
		const int height = 2160;
		var pixels = CreateTestImage(width, height);
		var singleTime = MeasureSingleThread(pixels, width, height, Iterations);
		var parallelTime = MeasureParallelCpu(pixels, width, height, Iterations);
		var cpuSpeedup = singleTime / parallelTime;
		var projectedGpuMs = singleTime.TotalMilliseconds / ExpectedGpuSpeedupOverSingleThread;
		Console.WriteLine($"4K image blur ({width}x{height} = {width * height} pixels, {Iterations} iterations):");
		Console.WriteLine($"  SingleThread: {singleTime.TotalMilliseconds:F2}ms");
		Console.WriteLine($"  ParallelCpu:  {parallelTime.TotalMilliseconds:F2}ms");
		Console.WriteLine($"  CPU Speedup:  {cpuSpeedup:F2}x");
		Console.WriteLine($"  Projected GPU: ~{projectedGpuMs:F2}ms ({ExpectedGpuSpeedupOverSingleThread:F0}x from CUDA reference)");
		Assert.That(cpuSpeedup, Is.GreaterThan(1.5),
			"CPU parallel should provide at least 1.5x speedup for 4K blur");
		Assert.That(projectedGpuMs, Is.LessThan(parallelTime.TotalMilliseconds),
			"Projected GPU time should beat CPU parallel based on CUDA reference data");
	}

	[Test]
	public void BlurComplexityMakesEvenMediumImagesWorthParallelizing()
	{
		const int width = 320;
		const int height = 320;
		var totalPixels = width * height;
		var blurComplexity = BrightnessPerformanceTests.EstimateComplexity(totalPixels,
			BlurBodyInstructionCount);
		Console.WriteLine($"320x320 blur: {totalPixels} pixels × {BlurBodyInstructionCount} " +
			$"body instructions = {blurComplexity} complexity");
		Assert.That(BrightnessPerformanceTests.ShouldParallelize(totalPixels,
			BlurBodyInstructionCount), Is.True,
			$"320x320 blur ({blurComplexity} complexity) should parallelize — " +
			"complex body compensates for moderate pixel count");
	}

	[Test]
	public void BlurVsBrightnessComplexityComparison()
	{
		const int pixels = 100_000;
		var brightnessComplexity = BrightnessPerformanceTests.EstimateComplexity(pixels,
			BrightnessBodyInstructionCount);
		var blurComplexity = BrightnessPerformanceTests.EstimateComplexity(pixels,
			BlurBodyInstructionCount);
		Console.WriteLine($"At {pixels} pixels:");
		Console.WriteLine($"  Brightness: {pixels} × {BrightnessBodyInstructionCount} = " +
			$"{brightnessComplexity} complexity → parallelize: " +
			$"{BrightnessPerformanceTests.ShouldParallelize(pixels, BrightnessBodyInstructionCount)}");
		Console.WriteLine($"  Blur 5x5:   {pixels} × {BlurBodyInstructionCount} = " +
			$"{blurComplexity} complexity → parallelize: " +
			$"{BrightnessPerformanceTests.ShouldParallelize(pixels, BlurBodyInstructionCount)}");
		Assert.That(blurComplexity, Is.GreaterThan(brightnessComplexity),
			"Blur should have higher complexity than brightness for same pixel count");
		Assert.That(BrightnessPerformanceTests.ShouldParallelize(pixels,
			BlurBodyInstructionCount), Is.True,
			"Blur should parallelize at 100K pixels due to complex body");
	}

	[TestCase(10)]
	[TestCase(20)]
	public void BlurProducesCorrectResults(int imageSize)
	{
		var pixels = CreateTestImage(imageSize, imageSize);
		var expected = new byte[pixels.Length];
		Array.Copy(pixels, expected, pixels.Length);
		ApplyBlurSingleThread(expected, imageSize, imageSize);
		var parallelResult = new byte[pixels.Length];
		Array.Copy(pixels, parallelResult, pixels.Length);
		ApplyBlurParallel(parallelResult, imageSize, imageSize);
		Assert.That(parallelResult, Is.EqualTo(expected),
			"Parallel blur result must match single-thread result byte-for-byte");
	}

	[Test]
	public void BlurIsMuchSlowerThanBrightnessShowingBodyComplexityMatters()
	{
		const int width = 1000;
		const int height = 1000;
		var pixels = CreateTestImage(width, height);
		var brightnessTime = MeasureBrightnessSingleThread(pixels, Iterations);
		var blurTime = MeasureSingleThread(pixels, width, height, Iterations);
		var ratio = blurTime / brightnessTime;
		Console.WriteLine($"1MP image single-thread ({Iterations} iterations):");
		Console.WriteLine($"  Brightness: {brightnessTime.TotalMilliseconds:F2}ms");
		Console.WriteLine($"  Blur 5x5:   {blurTime.TotalMilliseconds:F2}ms");
		Console.WriteLine($"  Blur/Brightness ratio: {ratio:F1}x slower");
		Assert.That(blurTime, Is.GreaterThan(brightnessTime),
			"5x5 blur should be significantly slower than simple brightness due to 25 neighbor reads");
	}

	/// <summary>
	/// Brightness: 3 channels × (read + add + clamp) ≈ 6 instructions per pixel.
	/// Blur 5×5: 25 neighbor reads + 25 additions + 1 division, per channel (×3) ≈ 30 instructions.
	/// </summary>
	private const int BlurBodyInstructionCount = 30;
	private const int BrightnessBodyInstructionCount = 6;

	/// <summary>
	/// From Strict.Compiler.Cuda.Tests/BlurPerformanceTests reference data (2048×1024, 200 iterations):
	/// SingleThread: 4594ms, CudaGpu: 32ms → 143x speedup. We use a conservative 50x for blur
	/// since blur has better GPU utilization than brightness (more ALU work per memory access).
	/// </summary>
	private const double ExpectedGpuSpeedupOverSingleThread = 50;

	private const int Iterations = 5;

	private static byte[] CreateTestImage(int width, int height)
	{
		var pixels = new byte[width * height * 3];
		var random = new Random(42);
		random.NextBytes(pixels);
		return pixels;
	}

	private static void ApplyBlurSingleThread(byte[] pixels, int width, int height)
	{
		var output = new byte[pixels.Length];
		var stride = width * 3;
		for (var row = 0; row < height; row++)
		for (var column = 0; column < width; column++)
		{
			if (row >= 2 && row < height - 2 && column >= 2 && column < width - 2)
				BlurInteriorPixel(pixels, output, row, column, stride);
			else
				BlurEdgePixel(pixels, output, row, column, width, height, stride);
		}
		Buffer.BlockCopy(output, 0, pixels, 0, pixels.Length);
	}

	private static void ApplyBlurParallel(byte[] pixels, int width, int height)
	{
		var output = new byte[pixels.Length];
		var stride = width * 3;
		Parallel.For(0, height, row =>
		{
			for (var column = 0; column < width; column++)
			{
				if (row >= 2 && row < height - 2 && column >= 2 && column < width - 2)
					BlurInteriorPixel(pixels, output, row, column, stride);
				else
					BlurEdgePixel(pixels, output, row, column, width, height, stride);
			}
		});
		Buffer.BlockCopy(output, 0, pixels, 0, pixels.Length);
	}

	private static void BlurInteriorPixel(byte[] source, byte[] output, int row, int column,
		int stride)
	{
		var baseIndex = row * stride + column * 3;
		for (var channel = 0; channel < 3; channel++)
		{
			var sum =
				source[baseIndex - 2 * stride - 6 + channel] +
				source[baseIndex - 2 * stride - 3 + channel] +
				source[baseIndex - 2 * stride + channel] +
				source[baseIndex - 2 * stride + 3 + channel] +
				source[baseIndex - 2 * stride + 6 + channel] +
				source[baseIndex - stride - 6 + channel] +
				source[baseIndex - stride - 3 + channel] +
				source[baseIndex - stride + channel] +
				source[baseIndex - stride + 3 + channel] +
				source[baseIndex - stride + 6 + channel] +
				source[baseIndex - 6 + channel] +
				source[baseIndex - 3 + channel] +
				source[baseIndex + channel] +
				source[baseIndex + 3 + channel] +
				source[baseIndex + 6 + channel] +
				source[baseIndex + stride - 6 + channel] +
				source[baseIndex + stride - 3 + channel] +
				source[baseIndex + stride + channel] +
				source[baseIndex + stride + 3 + channel] +
				source[baseIndex + stride + 6 + channel] +
				source[baseIndex + 2 * stride - 6 + channel] +
				source[baseIndex + 2 * stride - 3 + channel] +
				source[baseIndex + 2 * stride + channel] +
				source[baseIndex + 2 * stride + 3 + channel] +
				source[baseIndex + 2 * stride + 6 + channel];
			output[baseIndex + channel] = (byte)(sum / 25);
		}
	}

	private static void BlurEdgePixel(byte[] source, byte[] output, int row, int column,
		int width, int height, int stride)
	{
		var baseIndex = row * stride + column * 3;
		for (var channel = 0; channel < 3; channel++)
		{
			var sum = 0;
			var count = 0;
			for (var kernelY = -2; kernelY <= 2; kernelY++)
			for (var kernelX = -2; kernelX <= 2; kernelX++)
			{
				var neighborX = column + kernelX;
				var neighborY = row + kernelY;
				if ((uint)neighborX >= (uint)width || (uint)neighborY >= (uint)height)
					continue;
				sum += source[neighborY * stride + neighborX * 3 + channel];
				count++;
			}
			output[baseIndex + channel] = (byte)(sum / count);
		}
	}

	private static TimeSpan MeasureSingleThread(byte[] sourcePixels, int width, int height,
		int iterations)
	{
		var pixels = new byte[sourcePixels.Length];
		Array.Copy(sourcePixels, pixels, sourcePixels.Length);
		var stopwatch = Stopwatch.StartNew();
		for (var iteration = 0; iteration < iterations; iteration++)
			ApplyBlurSingleThread(pixels, width, height);
		stopwatch.Stop();
		return stopwatch.Elapsed;
	}

	private static TimeSpan MeasureParallelCpu(byte[] sourcePixels, int width, int height,
		int iterations)
	{
		var pixels = new byte[sourcePixels.Length];
		Array.Copy(sourcePixels, pixels, sourcePixels.Length);
		var stopwatch = Stopwatch.StartNew();
		for (var iteration = 0; iteration < iterations; iteration++)
			ApplyBlurParallel(pixels, width, height);
		stopwatch.Stop();
		return stopwatch.Elapsed;
	}

	private static void AdjustPixelBrightness(byte[] pixels, int pixelIndex)
	{
		for (var channel = 0; channel < 3; channel++)
		{
			var index = pixelIndex * 3 + channel;
			pixels[index] = (byte)Math.Clamp(pixels[index] + 10, 0, 255);
		}
	}

	private static TimeSpan MeasureBrightnessSingleThread(byte[] sourcePixels, int iterations)
	{
		var pixels = new byte[sourcePixels.Length];
		Array.Copy(sourcePixels, pixels, sourcePixels.Length);
		var pixelCount = pixels.Length / 3;
		var stopwatch = Stopwatch.StartNew();
		for (var iteration = 0; iteration < iterations; iteration++)
			for (var pixelIndex = 0; pixelIndex < pixelCount; pixelIndex++)
				AdjustPixelBrightness(pixels, pixelIndex);
		stopwatch.Stop();
		return stopwatch.Elapsed;
	}
}
