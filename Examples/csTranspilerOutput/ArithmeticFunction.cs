namespace SourceGeneratorTests;

public class ArithmeticFunction
{
	private List<int> numbers = new List<int>();
	public int Calculate(string operation)
	{
		if (operation == "add")
			return numbers[0] + numbers[1];
		if (operation == "subtract")
			return numbers[0] - numbers[1];
		if (operation == "multiply")
			return numbers[0] * numbers[1];
		if (operation == "divide")
			return numbers[0] / numbers[1];
	}

	[Test]
	public void CalculateTest()
	{
		Assert.That(() => new ArithmeticFunction(10, 5).Calculate("add") == 15));
		Assert.That(() => new ArithmeticFunction(10, 5).Calculate("subtract") == 5));
		Assert.That(() => new ArithmeticFunction(10, 5).Calculate("multiply") == 50));
	}
}