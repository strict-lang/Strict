using NUnit.Framework;

namespace Strict.Compiler.Cuda.Tests
{
    [Category("Slow")]
    public class KernelTests
    {
        [Test]
        public void CompileCudaCode() => new Kernel().CompileKernelAndSaveAsPtxFile();
    }
}