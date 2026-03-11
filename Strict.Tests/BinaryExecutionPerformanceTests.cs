using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Running;
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
	public void Setup()
	{
		rememberConsoleOut = Console.Out;
		Console.SetOut(TextWriter.Null);
		new Runner(TestPackage.Instance, StrictFilePath).Run().Dispose();
		var deserializer = new BytecodeDeserializer(BinaryFilePath, TestPackage.Instance);
		binaryPackage = deserializer.Package;
		instructions = deserializer.Instructions.Values.First();
		vm = new VirtualMachine(binaryPackage, deserializer.PrecompiledMethods);
	}

	private TextWriter rememberConsoleOut = null!;
	private VirtualMachine vm = null!;
	private List<Instruction> instructions = null!;
	private Package binaryPackage = null!;
	private static string StrictFilePath =>
		Path.Combine(AppContext.BaseDirectory, "Examples", "SimpleCalculator.strict");
	private static readonly string BinaryFilePath =
		Path.ChangeExtension(StrictFilePath, BytecodeSerializer.Extension);

	[TearDown]
	public void RestoreConsole()
	{
		Console.SetOut(rememberConsoleOut);
		binaryPackage.Dispose();
	}

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