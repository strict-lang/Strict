using NUnit.Framework;

namespace Strict.Compiler.Cuda.Tests
{
	/// <summary>
	/// Blurring a huge image might be a much better performance optization opportunity
	/// SingleThread() * 26214400: 549ms
	/// SingleThreadChunks() * 26214400: 264ms
	/// ParallelCpu() * 26214400: 107ms
	/// ParallelCpuChunks() * 26214400: 42ms
	/// </summary>
	[Category("Slow")]
	public class BlurPerformanceTests
	{
		[Test]
		public void CpuAndGpuLoops()
		{
			new TestPerformance(512*512*100, 100, Blur).Run();
		}

		private void Blur(int start, int chunkSize)
		{
			if (start < Width)
				return;
			for (int n = start; n < start + chunkSize; n++)
			{
				int output = image[n % Size] + image[(n - Width) % Size] + image[(n - 1) % Size] +
					image[(n + 1) % Size] + image[(n + Width) % Size];
				image[n % Size] = (byte)(output / 5);
			}
		}

		private byte[] image = new byte[Size];
		private const int Size = Width * 512;
		private const int Width = 512;
	}
}