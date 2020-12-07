using ManagedCuda;
using ManagedCuda.NVRTC;
using NUnit.Framework;

namespace Strict.Compiler.Cuda.Tests
{
	/// <summary>
	/// Blurring a huge image might be a much better performance optization opportunity
	///512x512:
	/// SingleThread() * 26214400: 549ms
	/// SingleThreadChunks() * 26214400: 264ms
	/// ParallelCpu() * 26214400: 107ms
	/// ParallelCpuChunks() * 26214400: 42ms
	/// CudaGpu() * 26214400: 16ms
	/// CudaGpuAndCpu() * 26214400: 13ms
	///2048*1024:
	/// SingleThread() * 209715200: 4594ms
	/// SingleThreadChunks() * 209715200: 2234ms
	/// ParallelCpu() * 209715200: 701ms
	/// ParallelCpuChunks() * 209715200: 296ms
	/// CudaGpu() * 209715200: 32ms
	/// CudaGpuAndCpu() * 209715200: 29ms
	/// </summary>
	[Category("Slow")]
	public class BlurPerformanceTests
	{
		[Test]
		public void CpuAndGpuLoops()
		{
			CompileKernel();
			new TestPerformance(Width*Height*BlurIterations, 100, Blur, BlurGpu).Run();
		}

		private const int Width = 2048;
		private const int Height = 1024;
		private const int BlurIterations = 100;

		public void CompileKernel()
		{
			//generate as output language obviously from strict code
			string code = @"extern ""C"" __global__ void blur(float *image, float *output, size_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > 2048 && tid < n) {
    output[tid] = (image[tid-2048]+image[tid-1]+image[tid]+image[tid+1]+image[tid+2048])/5;
  }
}";
			nvrtcResult result;
			using (var rtc = new CudaRuntimeCompiler(code, "blur"))
			{
				try
				{
					// see http://docs.nvidia.com/cuda/nvrtc/index.html for usage and options
					//https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
					//nvcc .\vectorAdd.cu -use_fast_math -ptx -m 64 -arch compute_61 -code sm_61 -o .\vectorAdd.ptx
					rtc.Compile(new[] { "--gpu-architecture=compute_61" });//TODO: use max capabilities on actual hardware we have at runtime
					result = nvrtcResult.Success;
				} catch(NVRTCException ex)
				{
					result = ex.NVRTCError;
					var r = ex.Message;
				}
				if (result == nvrtcResult.Success)
				{
					int deviceID = 0;
					CudaContext ctx = new CudaContext(deviceID);
					kernel = ctx.LoadKernelPTX(rtc.GetPTX(), "blur");
					kernel.GridDimensions = (N + 511) / 512;
					kernel.BlockDimensions = 512;
					float[] input = new float[N];
					d_A = input;
					d_C = new CudaDeviceVariable<float>(N);
				}
			}
		}

		private const int N = Width * Height;
		private CudaKernel kernel;
		private CudaDeviceVariable<float> d_A;
		private CudaDeviceVariable<float> d_C;
		
		private void Blur(int start, int chunkSize)
		{
			if (start < Width)
				return;
			const int Size = N;
			for (int n = start; n < start + chunkSize; n++)
			{
				int output = image[n % Size] + image[(n - Width) % Size] + image[(n - 1) % Size] +
					image[(n + 1) % Size] + image[(n + Width) % Size];
				image[n % Size] = (byte)(output / 5);
			}
		}

		private byte[] image = new byte[Width * Height];

		private void BlurGpu(int iterations)
		{
			for (int i = 0; i < iterations / N; i++)
				kernel.Run(d_A.DevicePointer, d_C.DevicePointer, N);
			// Copy result from device memory to host memory
			// h_C contains the result in host memory
			float[] h_C = d_C;
		}
	}
}