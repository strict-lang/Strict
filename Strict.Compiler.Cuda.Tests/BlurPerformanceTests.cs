using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using ManagedCuda;
using ManagedCuda.NVRTC;
using NUnit.Framework;

namespace Strict.Compiler.Cuda.Tests;

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
/// Strict should be faster than all of these
/// </summary>
[Category("Slow")]
public class BlurPerformanceTests
{
	//ncrunch: no coverage start
	[Test]
	public void CpuAndGpuLoops()
	{
		//both break the image pretty badly, but it is still somehow there .. needs better code yo
		LoadImage();
		CompileKernel();
		new TestPerformance(Width * Height * BlurIterations, 100, Blur, BlurGpu, SaveImage).Run();
	}

	private void LoadImage()
	{
		var bitmap =
			new Bitmap(@"TexturedMeshTests.RenderTexturedBoxPlaneAndSphereWithImage.approved.png");
		Width = bitmap.Width;
		Height = bitmap.Height;
		var data = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height),
			ImageLockMode.ReadOnly, bitmap.PixelFormat);
		CreateColorImageFromBitmapData(bitmap, data);
	}

	private unsafe void CreateColorImageFromBitmapData(Bitmap bitmap, BitmapData data)
	{
		image = new byte[Width * Height * 4];
		var pointer = (byte*)data.Scan0;
		var offsetIncrease = bitmap.PixelFormat is PixelFormat.Format24bppRgb
			? 3
			: 4;
		for (var y = 0; y < Height; y++)
		for (var x = 0; x < Width; x++)
		{
			image[(x + y * Width) * 4] = *(pointer + 2);
			image[(x + y * Width) * 4 + 1] = *(pointer + 1);
			image[(x + y * Width) * 4 + 2] = *(pointer + 0);
			pointer += offsetIncrease;
		}
	}

	private int Width;
	private int Height;
	private byte[] image = Array.Empty<byte>();
	private const int BlurIterations = 200;

	// ReSharper disable once MethodTooLong
	public void CompileKernel()
	{
		//generate as output language obviously from strict code
		const string Code = @"extern ""C"" __global__ void blur(unsigned char* image, unsigned char* output, size_t width, size_t height)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > width && tid < width*height-width) {
    output[tid] = image[tid];// (image[tid-2048]+image[tid-1]+image[tid]+image[tid+1]+image[tid+2048])/5;
  }
}";
		using var rtc = new CudaRuntimeCompiler(Code, "blur");
		try
		{
			// Use max capabilities on actual hardware we have at runtime
			var computeVersion = CudaContext.GetDeviceComputeCapability(0);
			var shaderModelVersion = "" + computeVersion.Major + computeVersion.Minor;
			Console.WriteLine("ShaderModelVersion=" + shaderModelVersion);
			// see http://docs.nvidia.com/cuda/nvrtc/index.html for usage and options
			//https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
			//nvcc .\vectorAdd.cu -use_fast_math -ptx -m 64 -arch compute_61 -code sm_61 -o .\vectorAdd.ptx
			//https://docs.nvidia.com/cuda/nvrtc/index.html#group__options
			rtc.Compile(new[] { "--gpu-architecture=compute_" + shaderModelVersion });
			Console.WriteLine("Cuda compile log: " + rtc.GetLogAsString());
			const int DeviceID = 0;
			var ctx = new CudaContext(DeviceID);
			kernel = ctx.LoadKernelPTX(rtc.GetPTX(), "blur");
			kernel.GridDimensions = (Size + 511) / 512;
			kernel.BlockDimensions = 512;
			//unused: float[] copyInput = new float[Size];
			input = image;
			output = new CudaDeviceVariable<byte>(Size);
		}
		catch (NVRTCException ex)
		{
			Console.WriteLine("Cuda compile log: " + rtc.GetLogAsString());
			throw new Exception(ex.NVRTCError + " " + ex);
		}
	}

	private int Size => Width * Height * 4;
	private CudaKernel kernel = null!;
	private CudaDeviceVariable<byte> input = null!;
	private CudaDeviceVariable<byte> output = null!;

	private void Blur(int start, int chunkSize)
	{
		if (start < Width * 4)
			return;
		var size = Size;
		for (var n = start; n < start + chunkSize; n++)
			// ReSharper disable once ComplexConditionExpression
			image[(n + 0) % size] = (byte)((image[n % size] + image[(n - Width * 4) % size] +
				image[(n - 4) % size] + image[(n + 4) % size] + image[(n + Width * 4) % size]) / 5);
	}

	private void BlurGpu(int iterations)
	{
		for (var i = 0; i < BlurIterations; i++)
			kernel.Run(input.DevicePointer, output.DevicePointer, Width, Height);
		// Copy result from device memory to host memory
		// h_C contains the result in host memory
		//float[] copyOutput = output;
		image = output;
	}

	private void SaveImage(string methodName)
	{
		var filePath = methodName + "Blurred.jpg";
		using var bitmap = AsBitmap(image);
		using Stream stream = File.Open(filePath, FileMode.Create, FileAccess.ReadWrite);
		bitmap.Save(stream, ImageFormat.Jpeg);
		// Load original image back so we start fresh for the next performance test
		LoadImage();
	}

	public unsafe Bitmap AsBitmap(byte[] data, bool flipTopBottom = false)
	{
		var bitmap = new Bitmap(Width, Height);
		var bitmapData = bitmap.LockBits(new Rectangle(0, 0, Width, Height), ImageLockMode.WriteOnly,
			PixelFormat.Format24bppRgb);
		var bitmapPointer = (byte*)bitmapData.Scan0.ToPointer();
		SwitchRgbToBgr(data, bitmapPointer, bitmapData.Stride, flipTopBottom);
		bitmap.UnlockBits(bitmapData);
		return bitmap;
	}

	private unsafe void SwitchRgbToBgr(byte[] data, byte* bitmapPointer, int stride,
		bool flipTopBottom)
	{
		for (var y = 0; y < Height; ++y)
		for (var x = 0; x < Width; ++x)
		{
			var targetIndex = y * stride + x * 3;
			// ReSharper disable once ComplexConditionExpression
			var sourceIndex = flipTopBottom
				? ((Height - 1 - y) * Width + x) * 3
				: (y * Width + x) * 3;
			bitmapPointer[targetIndex] = data[sourceIndex + 2];
			bitmapPointer[targetIndex + 1] = data[sourceIndex + 1];
			bitmapPointer[targetIndex + 2] = data[sourceIndex];
		}
	}
}