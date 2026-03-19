using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Running;
using Strict.Bytecode;
using Strict.Bytecode.Serialization;
using Strict.Language;
using Strict.Language.Tests;

namespace Strict.Tests;

[MemoryDiagnoser]
[SimpleJob(RunStrategy.Throughput, warmupCount: 1, iterationCount: 10)]
public class BinaryExecutionPerformanceTests
{
	[SetUp]
	public void SetupConsole()
	{
		rememberConsoleOut = Console.Out;
		Console.SetOut(TextWriter.Null);
	}

	private TextWriter rememberConsoleOut = null!;

	[TearDown]
	public void RestoreConsole() => Console.SetOut(rememberConsoleOut);

	public async Task<VirtualMachine> CreateVm()
	{
		try
		{
			await new Runner(StrictFilePath).Run();
		}
		//ncrunch: no coverage start
		catch (IOException)
		{
			// Try again if the file was used in another test
			Thread.Sleep(100);
			await new Runner(StrictFilePath).Run();
		} //ncrunch: no coverage end
		var executable = new BinaryExecutable(BinaryFilePath, TestPackage.Instance);
		return new VirtualMachine(executable);
	}

	private static string StrictFilePath => RunnerTests.GetExamplesFilePath("SimpleCalculator");
	private static readonly string BinaryFilePath =
		Path.ChangeExtension(StrictFilePath, BinaryExecutable.Extension);

	[Test]
	public async Task ExecuteBinaryOnce()
	{
		var vm = await CreateVm();
		vm.ExecuteRun();
	}

	[Test]
	public async Task ExecuteBinary1000Times()
	{
		var vm = await CreateVm();
		for (var run = 0; run < 1000; run++)
			vm.ExecuteRun();
	}

	//ncrunch: no coverage start
	[Benchmark]
	public async Task ExecuteBinary() => (await CreateVm()).ExecuteRun();

	[Test]
	[Category("Manual")]
	public void BenchmarkCompare() => BenchmarkRunner.Run<BinaryExecutionPerformanceTests>();
}