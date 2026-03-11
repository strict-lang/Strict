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
		EnsureBinaryFileExists();
		var deserializer = new BytecodeDeserializer(BinaryFilePath, TestPackage.Instance);
		binaryPackage = deserializer.Package;
		instructions = deserializer.Instructions.Values.First();
		vm = new VirtualMachine(binaryPackage);
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

	private static void EnsureBinaryFileExists()
	{
		if (!File.Exists(BinaryFilePath))
			new Runner(TestPackage.Instance, StrictFilePath).Run().Dispose(); //ncrunch: no coverage
	}

	[Test]
	[Benchmark]
	public void ExecuteBinaryOnce() => vm.Execute(instructions);

	[Test]
	public void ExecuteBinaryThousandTimes()
	{
		for (var run = 0; run < 1000; run++)
			vm.Execute(instructions);
	}

	//ncrunch: no coverage start
	[Test]
	[Category("Manual")]
	public void BenchmarkCompare() => BenchmarkRunner.Run<BinaryExecutionPerformanceTests>();
}
