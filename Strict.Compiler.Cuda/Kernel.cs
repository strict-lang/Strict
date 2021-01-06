using System.IO;
using ManagedCuda;
using ManagedCuda.CudaBlas;
using ManagedCuda.NVRTC;

namespace Strict.Compiler.Cuda
{
	public class Kernel
	{
		public Kernel() => Context = new CudaContext(0);
		public CudaContext Context { get; init; }
		public CudaBlas Handle => handle ??= new();
		private CudaBlas? handle;

		public void Dispose()
		{
			Context.Dispose();
			handle?.Dispose();
		}
		
		public void CompileKernelAndSaveAsPtxFile()
		{
			//generate as output language obviously from strict code
			string code = @"extern ""C"" __global__ void saxpy(float a, float *x, float *y, float *out, size_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    out[tid] = a * x[tid] + y[tid];
  }
}";
			nvrtcResult result;
			using var rtc = new CudaRuntimeCompiler(code, "saxpy");
			try
			{
				// see http://docs.nvidia.com/cuda/nvrtc/index.html for usage and options
				//https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
				//nvcc .\vectorAdd.cu -use_fast_math -ptx -m 64 -arch compute_61 -code sm_61 -o .\vectorAdd.ptx
				rtc.Compile(new[] { "--gpu-architecture=compute_61" });
				result = nvrtcResult.Success;
			}
			catch(NVRTCException ex)
			{
				result = ex.NVRTCError;
			}
			if (result == nvrtcResult.Success)
			{
				//we could consume right away, this could be done in for caching or in the background for next usage
				using var file = new StreamWriter(@"convokernel.ptx");
				file.Write(rtc.GetPTXAsString());
			}
		}
		
		public void Example()
		{
			//see above, grab NVRTC compiler code and do this all dynamically
			int N = 50000;
			int deviceID = 0;
			CudaContext ctx = new CudaContext(deviceID);
			CudaKernel kernel = ctx.LoadKernel("VectorAdd.ptx", "VecAdd");
			kernel.GridDimensions = (N + 255) / 256;
			kernel.BlockDimensions = 256;

			// Allocate input vectors h_A and h_B in host memory
			float[] h_A = new float[N];
			float[] h_B = new float[N];

			// Initialize input vectors h_A, h_B

			// Allocate vectors in device memory and copy vectors from host memory to device memory 
			CudaDeviceVariable<float> d_A = h_A;
			CudaDeviceVariable<float> d_B = h_B;
			CudaDeviceVariable<float> d_C = new CudaDeviceVariable<float>(N);

			// Invoke kernel
			kernel.Run(d_A.DevicePointer, d_B.DevicePointer, d_C.DevicePointer, N);

			// Copy result from device memory to host memory
			// h_C contains the result in host memory
			//float[] h_C = d_C;
		}
	}
}
