using ManagedCuda;
using ManagedCuda.CudaBlas;
using ManagedCuda.NVRTC;

namespace Strict.Compiler.Cuda;

public class Kernel : IDisposable
{
	public CudaContext Context { get; } = new(0);
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
		const string Code = @"extern ""C"" __global__ void saxpy(float a, float *x, float *y, float *out, size_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    out[tid] = a * x[tid] + y[tid];
  }
}";
		using var rtc = new CudaRuntimeCompiler(Code, "saxpy");
		var result = CompileInTryCatch(rtc);
		if (result == nvrtcResult.Success)
			GeneratePtxOutputFile(rtc);
	}

	private static nvrtcResult CompileInTryCatch(CudaRuntimeCompiler rtc)
	{
		nvrtcResult result;
		try
		{
			// see http://docs.nvidia.com/cuda/nvrtc/index.html for usage and options
			//https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
			//nvcc .\vectorAdd.cu -use_fast_math -ptx -m 64 -arch compute_61 -code sm_61 -o .\vectorAdd.ptx
			rtc.Compile(["--gpu-architecture=compute_61"]);
			result = nvrtcResult.Success;
		}
		catch (NVRTCException ex)
		{
			result = ex.NVRTCError;
		}
		return result;
	}

	private static void GeneratePtxOutputFile(CudaRuntimeCompiler rtc)
	{ //we could consume right away, this could be done in for caching or in the background for next usage
		using var file = new StreamWriter("VectorAdd.ptx");
		file.Write(rtc.GetPTXAsString());
	}
}