namespace Strict.Runtime.Tests;

public class BytecodeInterpreterTestsKata : BytecodeInterpreterTests
{
	[Test]
	public void BestTimeToBuyStocksKata()
	{
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource("Stock",
			"Stock(7, 1, 5, 3, 6, 4).MaxProfit",
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
		Assert.That(vm.Execute(statements).Returns?.Value, Is.EqualTo(5));
	}

	[TestCase("RemoveParentheses(\"some(thing)\").Remove", "some")]
	[TestCase("RemoveParentheses(\"(some)thing\").Remove", "thing")]
	public void RemoveParentheses(string methodCall, string expectedResult)
	{
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource("RemoveParentheses",
			methodCall, RemoveParenthesesKata)).Generate();
		Assert.That(vm.Execute(statements).Returns?.Value, Is.EqualTo(expectedResult));
	}

	[TestCase("Invertor(1, 2, 3, 4, 5).Invert", "-1-2-3-4-5")]
	public void InvertValues(string methodCall, string expectedResult)
	{
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource("Invertor",
			methodCall, InvertValueKata)).Generate();
		Assert.That(vm.Execute(statements).Returns?.Value, Is.EqualTo(expectedResult));
	}

	[Test]
	public void CountingSheepKata()
	{
		var statements = new ByteCodeGenerator(GenerateMethodCallFromSource("SheepCounter",
			"SheepCounter(true, true, true, false, true, true, true, true, true, false, true, false, true, false, false, true, true, true, true, true, false, false, true, true).Count",
			"has sheep Booleans",
			"Count Number",
			"\tmutable result = 0",
			"\tfor sheep",
			"\t\tif value",
			"\t\t\tresult.Increment",
			"\tresult")).Generate();
		Assert.That(vm.Execute(statements).Returns!.Value, Is.EqualTo(17));
	}
}