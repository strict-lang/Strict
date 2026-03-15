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
	public void Setup()
	{
		rememberConsoleOut = Console.Out;
		Console.SetOut(TextWriter.Null);
		if (!File.Exists(BinaryFilePath))
			new Runner(StrictFilePath).Run().Dispose(); //ncrunch: no coverage
		var bytecodeTypes = new BytecodeDeserializer(BinaryFilePath).Deserialize(TestPackage.Instance);
		binaryPackage = TestPackage.Instance;
		instructions = bytecodeTypes.Find("SimpleCalculator", "Run", 0) ?? new List<Instruction>();
		vm = new VirtualMachine(binaryPackage);
	}

	private TextWriter rememberConsoleOut = null!;
	private VirtualMachine vm = null!;
	private List<Instruction> instructions = null!;
	private Package binaryPackage = null!;
	private static string StrictFilePath => RunnerTests.GetExamplesFilePath("SimpleCalculator");
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