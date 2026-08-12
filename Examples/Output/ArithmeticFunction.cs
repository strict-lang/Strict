namespace SourceGeneratorTests;

public class ArithmeticFunction
{
	private int first;
	private int second;
	public int Calculate(string operation)
	{
		switch (operation)
		{
			case "add": return first + second;
			case "subtract": return first - second;
			case "multiply": return first * second;
			case "divide": return first / second;
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
