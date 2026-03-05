using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Running;

namespace Strict.Runtime.Tests;

[MemoryDiagnoser]
[SimpleJob(RunStrategy.Throughput, warmupCount: 1, iterationCount: 10)]
public class AdderProgramTests : BaseVirtualMachineTests
{
	[SetUp]
	public void Setup() => vm = new BytecodeInterpreter();

	private BytecodeInterpreter vm = null!;
	private static readonly string[] AdderProgramCode =
	[
		"has numbers",
		"AddTotals Numbers",
		"\tmutable total = 0",
		"\tmutable results = Numbers",
		"\tfor numbers",
		"\t\ttotal = total + value",
		"\t\tresults.Add(total)",
		"\tresults"
	];

	private List<decimal> ExecuteAddTotals(string methodCall)
	{
		var result = vm.Execute(
			new ByteCodeGenerator(GenerateMethodCallFromSource("AdderProgram",
				methodCall, AdderProgramCode)).Generate()).Returns!.Value;
		return result.List.Items.Select(item => (decimal)item.Number).ToList();
	}

	[Test]
	public void AddTotalsForSingleNumber() =>
		Assert.That(ExecuteAddTotals("AdderProgram(5).AddTotals"), Is.EqualTo(new[] { 5m }));

	[Test]
	public void AddTotalsForTwoNumbers() =>
		Assert.That(ExecuteAddTotals("AdderProgram(1, 2).AddTotals"), Is.EqualTo(new[] { 1m, 3m }));

	//ncrunch: no coverage start
	[Test]
	[Category("Slow")]
	[Benchmark]
	public void AddTotalsForThreeNumbers() =>
		Assert.That(ExecuteAddTotals("AdderProgram(1, 2, 3).AddTotals"),
			Is.EqualTo(new[] { 1m, 3m, 6m }));

	[Test]
	[Category("Manual")]
	public void BenchmarkCompare() => BenchmarkRunner.Run<AdderProgramTests>();
}
