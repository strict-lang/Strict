using NUnit.Framework;

namespace Strict.Compiler.Cuda.Tests
{
	public class KernelTests
	{
		[Test][Category("Slow")]
		public void CompileCudaCode()
		{
			//needs around 300ms
			new Kernel().CompileKernelAndSaveAsPtxFile();
		}
	}
}