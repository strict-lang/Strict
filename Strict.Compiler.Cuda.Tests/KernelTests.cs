using NUnit.Framework;

namespace Strict.Compiler.Cuda.Tests;

public class KernelTests
{
	[Test]
	public void CompileCudaCode()
	{
		using var kernel = new Kernel();
		kernel.CompileKernelAndSaveAsPtxFile();
	}
}