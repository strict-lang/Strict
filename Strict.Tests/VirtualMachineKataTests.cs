using Strict.Bytecode;
using Strict.Bytecode.Tests;
using Strict.Language.Tests;

namespace Strict.Tests;

public sealed class BytecodeInterpreterKataTests : TestBytecode
{
	[SetUp]
	public void Setup() => vm = new VirtualMachine(TestPackage.Instance);

	private VirtualMachine vm = null!;

	[Test]
	public void BestTimeToBuyStocksKata()
	{
		var instructions = new BytecodeGenerator(GenerateMethodCallFromSource("Stock",
			"Stock(7, 1, 5, 3, 6, 4).MaxProfit",
			// @formatter:off
			"has prices Numbers",
			"MaxProfit Number",
			"\tmutable min = 10000000",
			"\tmutable max = 0",
			"\tfor prices",
			"\t\tif value < min",
			"\t\t\tmin = value",
			"\t\telse if value - min > max",
			"\t\t\tmax = value - min",
			"\tmax")).Generate();
		// @formatter:on
		Assert.That(vm.Execute(instructions).Returns!.Value.Number, Is.EqualTo(5));
	}

	[TestCase("RemoveParentheses(\"some(thing)\").Remove", "some")]
	[TestCase("RemoveParentheses(\"(some)thing\").Remove", "thing")]
	public void RemoveParentheses(string methodCall, string expectedResult)
	{
		var instructions = new BytecodeGenerator(GenerateMethodCallFromSource("RemoveParentheses",
			methodCall,
			// @formatter:off
			"has text",
			"Remove Text",
			"\tmutable result = \"\"",
			"\tmutable count = 0",
			"\tfor text",
			"\t\tif value is \"(\"",
			"\t\t\tcount = count + 1",
			"\t\tif count is 0",
			"\t\t\tresult = result + value",
			"\t\tif value is \")\"",
			"\t\t\tcount = count - 1",
			"\tresult")).Generate();
		// @formatter:on
		Assert.That(vm.Execute(instructions).Returns!.Value.Text, Is.EqualTo(expectedResult));
	}

	[TestCase("Invertor(1, 2, 3, 4, 5).Invert", "-1-2-3-4-5")]
	public void InvertValues(string methodCall, string expectedResult)
	{
		var instructions = new BytecodeGenerator(GenerateMethodCallFromSource("Invertor", methodCall,
			// @formatter:off
			"has numbers",
			"Invert Text",
			"\tmutable result = \"\"",
			"\tfor numbers",
			"\t\tresult = result + value * -1",
			"\tresult")).Generate();
		// @formatter:on
		Assert.That(vm.Execute(instructions).Returns!.Value.Text, Is.EqualTo(expectedResult));
	}

	[Test]
	public void CountingSheepKata()
	{
		var instructions = new BytecodeGenerator(GenerateMethodCallFromSource("SheepCounter",
			"SheepCounter(true, true, true, false, true, true, true, true, true, false, true, false, " +
			"true, false, false, true, true, true, true, true, false, false, true, true).Count",
			// @formatter:off
			"has sheep Booleans",
			"Count Number",
			"\tmutable result = 0",
			"\tfor sheep",
			"\t\tif value",
			"\t\t\tresult.Increment",
			"\tresult")).Generate();
		// @formatter:on
		Assert.That(vm.Execute(instructions).Returns!.Value.Number, Is.EqualTo(17));
	}
}