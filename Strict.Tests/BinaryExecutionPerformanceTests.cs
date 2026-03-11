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
	private VirtualMachine vm = null!;
	private List<Instruction> instructions = null!;
	private Package binaryPackage = null!;
	private static readonly string BinaryFilePath =
		Path.ChangeExtension(
			Path.Combine(AppContext.BaseDirectory, "Examples", "SimpleCalculator.strict"),
			BytecodeSerializer.Extension)!;

	[GlobalSetup]
	[SetUp]
	public void Setup()
	{
		EnsureBinaryFileExists();
		(binaryPackage, var instructionsByType) =
			BytecodeSerializer.LoadTypesAndDeserializeAll(BinaryFilePath, TestPackage.Instance);
		instructions = instructionsByType.Values.First();
		vm = new VirtualMachine(binaryPackage);
	}

	private static void EnsureBinaryFileExists()
	{
		if (!File.Exists(BinaryFilePath))
			RunSilently(() => new Runner(TestPackage.Instance,
				Path.Combine(AppContext.BaseDirectory, "Examples", "SimpleCalculator.strict")).Run().Dispose());
	}

	[Test]
	[Benchmark]
	public void ExecuteBinaryOnce() => RunSilently(() => vm.Execute(instructions));

	private static void RunSilently(Action action)
	{
		var saved = Console.Out;
		Console.SetOut(TextWriter.Null);
		try
		{
			action();
		}
		finally
		{
			Console.SetOut(saved);
		}
	}

	//ncrunch: no coverage start
	[Test]
	[Category("Slow")]
	public void ExecuteBinaryThousandTimes()
	{
		const int Runs = 1000;
		for (var run = 0; run < Runs; run++)
			ExecuteBinaryOnce();
	}

	[Test]
	[Category("Manual")]
	public void BenchmarkCompare() => BenchmarkRunner.Run<BinaryExecutionPerformanceTests>();
}
