namespace Strict.Runtime.Tests;

public sealed class AdderProgramTests : BaseVirtualMachineTests
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

	private List<decimal> ExecuteAddTotals(string methodCall) =>
		((IEnumerable<Expression>)vm.Execute(
			new ByteCodeGenerator(GenerateMethodCallFromSource("AdderProgram",
				methodCall, AdderProgramCode)).Generate()).Returns!.Value!).
		Select(e => Convert.ToDecimal(((Value)e).Data)).ToList();

	[Test]
	public void AddTotalsForSingleNumber() =>
		Assert.That(ExecuteAddTotals("AdderProgram(5).AddTotals"), Is.EqualTo(new[] { 5m }));

	[Test]
	public void AddTotalsForTwoNumbers() =>
		Assert.That(ExecuteAddTotals("AdderProgram(1, 2).AddTotals"), Is.EqualTo(new[] { 1m, 3m }));

	[Test]
	public void AddTotalsForThreeNumbers() =>
		Assert.That(ExecuteAddTotals("AdderProgram(1, 2, 3).AddTotals"),
			Is.EqualTo(new[] { 1m, 3m, 6m }));
}
