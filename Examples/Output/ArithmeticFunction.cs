namespace SourceGeneratorTests;

public class ArithmeticFunction
{
	private List<int> numbers = new List<int>();
	public int Calculate(string operation)
	{
		foreach (var index in numbers)
			switch (operation)
			{
				case "add": return value;
				case "subtract": return value;
				case "multiply": return value;
				case "divide": return value;
			}
	}

	[Test]
	public void CalculateTest()
	{
		Assert.That(() => new ArithmeticFunction(10, 5).Calculate("add") == 15));
		Assert.That(() => new ArithmeticFunction(10, 5).Calculate("subtract") == 5));
		Assert.That(() => new ArithmeticFunction(10, 5).Calculate("multiply") == 50));
	}
}