using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Running;
using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Language;
using Strict.Language.Tests;

namespace Strict.Tests;

[MemoryDiagnoser]
[SimpleJob(RunStrategy.Throughput, warmupCount: 1, iterationCount: 10)]
public class BinaryExecutionPerformanceTests
{
	[GlobalSetup]
	[SetUp]
	public async Task SetupAsync()
	{
		rememberConsoleOut = Console.Out;
		Console.SetOut(TextWriter.Null);
		if (File.Exists(BinaryFilePath))
			File.Delete(BinaryFilePath);
		await new Runner(StrictFilePath).Run(); //ncrunch: no coverage
		var executable = new BinaryExecutable(BinaryFilePath, TestPackage.Instance);
		instructions = executable.FindInstructions("SimpleCalculator", "Run", 0) ?? [];
		vm = new VirtualMachine(executable);
	}

	private TextWriter rememberConsoleOut = null!;
	private VirtualMachine vm = null!;
	private IReadOnlyList<Instruction> instructions = null!;
	private static string StrictFilePath => RunnerTests.GetExamplesFilePath("SimpleCalculator");
	private static readonly string BinaryFilePath =
		Path.ChangeExtension(StrictFilePath, BytecodeSerializer.Extension);

	[TearDown]
	public void RestoreConsole() => Console.SetOut(rememberConsoleOut);

	[Benchmark]
	public void ExecuteBinary() => vm.Execute(instructions); //ncrunch: no coverage

	[Test]
	public void ExecuteBinary1000Times()
	{
		for (var run = 0; run < 1000; run++)
			vm.Execute(instructions);
	}

	//ncrunch: no coverage start
	[Test]
	[Category("Manual")]
	public void BenchmarkCompare() => BenchmarkRunner.Run<BinaryExecutionPerformanceTests>();
}