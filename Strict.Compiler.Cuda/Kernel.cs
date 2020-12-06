using ManagedCuda;
using ManagedCuda.CudaBlas;

namespace Strict.Compiler.Cuda
{
	public class Kernel
	{
		public Kernel() => Context = new CudaContext(0);

		public CudaContext Context { get; init;}
		public CudaBlas Handle => handle ??= new();
		private CudaBlas? handle;

		public void Dispose()
		{
			Context.Dispose();
			if (Handle != null)
				Handle.Dispose();
		}
		
		public void Example()
		{
			//grab NVRTC compiler code and do this all dynamically
			int N = 50000;
			int deviceID = 0;
			CudaContext ctx = new CudaContext(deviceID);
			CudaKernel kernel = ctx.LoadKernel("VectorAdd.ptx", "VecAdd");
			kernel.GridDimensions = (N + 255) / 256;
			kernel.BlockDimensions = 256;

			// Allocate input vectors h_A and h_B in host memory
			float[] h_A = new float[N];
			float[] h_B = new float[N];

			// TODO: Initialize input vectors h_A, h_B

			// Allocate vectors in device memory and copy vectors from host memory to device memory 
			CudaDeviceVariable<float> d_A = h_A;
			CudaDeviceVariable<float> d_B = h_B;
			CudaDeviceVariable<float> d_C = new CudaDeviceVariable<float>(N);

			// Invoke kernel
			kernel.Run(d_A.DevicePointer, d_B.DevicePointer, d_C.DevicePointer, N);

			// Copy result from device memory to host memory
			// h_C contains the result in host memory
			float[] h_C = d_C;
		}
	}
}
