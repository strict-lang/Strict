using NUnit.Framework;

namespace Strict.Compiler.Cuda.Tests;

[Category("Manual")]
public class KernelTests
{
	[Test]
	public void CompileCudaCode() => new Kernel().CompileKernelAndSaveAsPtxFile();
}