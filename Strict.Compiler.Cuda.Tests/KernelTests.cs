using NUnit.Framework;

namespace Strict.Compiler.Cuda.Tests;

[Category("Manual")]
[Ignore("Requires GPU hardware and CUDA drivers")]
public class KernelTests
{
	[Test]
	public void CompileCudaCode() => new Kernel().CompileKernelAndSaveAsPtxFile();
}