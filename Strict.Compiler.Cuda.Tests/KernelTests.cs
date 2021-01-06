using NUnit.Framework;

namespace Strict.Compiler.Cuda.Tests
{
	[Category("Slow")]
	public class KernelTests
	{
		[Test]
		public void CompileCudaCode()
		{
			//needs around 300ms
			new Kernel().CompileKernelAndSaveAsPtxFile();
		}
	}
}