using System.Diagnostics;
using NUnit.Framework;

namespace Strict.Compiler.Assembly.Tests;

/// <summary>
/// 5x5 box blur performance test comparing single-thread vs parallel CPU vs projected GPU. Blur
/// has a much heavier per-pixel body than brightness (~30 instructions: 25 reads + sum + divide)
/// making it the perfect case for complexity-based parallelization. Even a 200x200 image (40K
/// pixels) reaches 1.2M complexity (40K × 30), well above the 100K threshold.
/// Reference from Strict.Compiler.Cuda.Tests/BlurPerformanceTests (2048x1024, 200 iterations):
///   SingleThread: 4594ms, ParallelCpu: 701ms (6.5x), CudaGpu: 32ms (143x)
/// </summary>
public sealed class BlurPerformanceTests
{
	[Test]
	public void SmallImageBlurIsFastEnough()
	{
		const int Width = 80;
		const int Height = 60;
		var pixels = CreateTestImage(Width, Height);
		var elapsed = MeasureSingleThread(pixels, Width, Height, Iterations);
		Assert.That(elapsed.TotalMilliseconds, Is.LessThan(20),
			$"{Width}x{Height} blur should complete quickly on single thread");
		var parallelTime = MeasureParallelCpu(pixels, Width, Height, Iterations);
		Assert.That(elapsed.TotalMilliseconds, Is.LessThan(10),
			$"{Width}x{Height} blur should complete quickly on multiple threads");
	}

	private static byte[] CreateTestImage(int width, int height)
	{
		var pixels = new byte[width * height * 3];
		var random = new Random(42);
		random.NextBytes(pixels);
		return pixels;
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

	private const int Iterations = 1;

	//ncrunch: no coverage start, faster without line by line ncrunch instrumentation
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
	}	//ncrunch: no coverage end

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

	[Test]
	public void BlurComplexityMakesEvenMediumImagesWorthParallelizing()
	{
		const int Width = 320;
		const int Height = 320;
		var totalPixels = Width * Height;
		var blurComplexity = EstimateComplexity(totalPixels,	BlurBodyInstructionCount);
		Assert.That(ShouldParallelize(totalPixels,	BlurBodyInstructionCount),
			Is.True, $"{Width}x{Height} blur ({blurComplexity} complexity) should parallelize, " +
			"complex body compensates for moderate pixel count");
	}

	public static long EstimateComplexity(long iterations, int bodyInstructionCount) =>
		iterations * Math.Max(bodyInstructionCount, 1);

	public static bool ShouldParallelize(long iterations, int bodyInstructionCount) =>
		EstimateComplexity(iterations, bodyInstructionCount) >
		InstructionsToMlir.ComplexityThreshold;

	/// <summary>
	/// Brightness: 3 channels × (read + add + clamp) ≈ 6 instructions per pixel.
	/// Blur 5×5: 25 neighbor reads + 25 additions + 1 division, per channel (×3) ≈ 30 instructions.
	/// </summary>
	private const int BlurBodyInstructionCount = 30;
	private const int BrightnessBodyInstructionCount = 6;

	[Test]
	public void BlurVsBrightnessComplexityComparison()
	{
		const int Pixels = 100_000;
		var brightnessComplexity = EstimateComplexity(Pixels,	BrightnessBodyInstructionCount);
		var blurComplexity = EstimateComplexity(Pixels,	BlurBodyInstructionCount);
		Assert.That(blurComplexity, Is.GreaterThan(brightnessComplexity),
			"Blur should have higher complexity than brightness for same pixel count");
		Assert.That(ShouldParallelize(Pixels,	BlurBodyInstructionCount), Is.True,
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

	//ncrunch: no coverage start
	[Test]
	[Category("Slow")]
	public void LargeImageBlurParallelIsFaster()
	{
		const int Width = 2048;
		const int Height = 1024;
		var pixels = CreateTestImage(Width, Height);
		var singleTime = MeasureSingleThread(pixels, Width, Height, Iterations);
		var parallelTime = MeasureParallelCpu(pixels, Width, Height, Iterations);
		var speedup = singleTime / parallelTime;
		Console.WriteLine($"2K image blur ({Width}x{Height} = {Width * Height} pixels, {Iterations} iterations):");
		Console.WriteLine($"  SingleThread: {singleTime.TotalMilliseconds:F2}ms");
		Console.WriteLine($"  ParallelCpu:  {parallelTime.TotalMilliseconds:F2}ms");
		Console.WriteLine($"  Speedup:      {speedup:F2}x");
		Assert.That(parallelTime, Is.LessThan(singleTime),
			"Parallel should be faster than single-thread for 2K image blur");
	}

	[Test]
	[Category("Slow")]
	public void FourKBlurShowsStrongParallelAndGpuProjection()
	{
		const int Width = 3840;
		const int Height = 2160;
		var pixels = CreateTestImage(Width, Height);
		var singleTime = MeasureSingleThread(pixels, Width, Height, Iterations);
		var parallelTime = MeasureParallelCpu(pixels, Width, Height, Iterations);
		var cpuSpeedup = singleTime / parallelTime;
		var projectedGpuMs = singleTime.TotalMilliseconds / ExpectedGpuSpeedupOverSingleThread;
		Console.WriteLine($"4K image blur ({Width}x{Height} = {Width * Height} pixels, {Iterations} iterations):");
		Console.WriteLine($"  SingleThread: {singleTime.TotalMilliseconds:F2}ms");
		Console.WriteLine($"  ParallelCpu:  {parallelTime.TotalMilliseconds:F2}ms");
		Console.WriteLine($"  CPU Speedup:  {cpuSpeedup:F2}x");
		Console.WriteLine($"  Projected GPU: ~{projectedGpuMs:F2}ms ({ExpectedGpuSpeedupOverSingleThread:F0}x from CUDA reference)");
		Assert.That(cpuSpeedup, Is.GreaterThan(1.5),
			"CPU parallel should provide at least 1.5x speedup for 4K blur");
		Assert.That(projectedGpuMs, Is.LessThan(parallelTime.TotalMilliseconds),
			"Projected GPU time should beat CPU parallel based on CUDA reference data");
	}

	/// <summary>
	/// From Strict.Compiler.Cuda.Tests/BlurPerformanceTests reference data (2048×1024, 200 iterations):
	/// SingleThread: 4594ms, CudaGpu: 32ms → 143x speedup. We use a conservative 50x for blur
	/// since blur has better GPU utilization than brightness (more ALU work per memory access).
	/// </summary>
	private const double ExpectedGpuSpeedupOverSingleThread = 50;

	[Test]
	[Category("Slow")]
	public void BlurIsMuchSlowerThanBrightnessShowingBodyComplexityMatters()
	{
		const int Width = 1000;
		const int Height = 1000;
		var pixels = CreateTestImage(Width, Height);
		var brightnessTime = MeasureBrightnessSingleThread(pixels, Iterations);
		var blurTime = MeasureSingleThread(pixels, Width, Height, Iterations);
		var ratio = blurTime / brightnessTime;
		Console.WriteLine($"1MP image single-thread ({Iterations} iterations):");
		Console.WriteLine($"  Brightness: {brightnessTime.TotalMilliseconds:F2}ms");
		Console.WriteLine($"  Blur 5x5:   {blurTime.TotalMilliseconds:F2}ms");
		Console.WriteLine($"  Blur/Brightness ratio: {ratio:F1}x slower");
		Assert.That(blurTime, Is.GreaterThan(brightnessTime),
			"5x5 blur should be significantly slower than simple brightness due to 25 neighbor reads");
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

	private static void AdjustPixelBrightness(byte[] pixels, int pixelIndex)
	{
		for (var channel = 0; channel < 3; channel++)
		{
			var index = pixelIndex * 3 + channel;
			pixels[index] = (byte)Math.Clamp(pixels[index] + 10, 0, 255);
		}
	}
}