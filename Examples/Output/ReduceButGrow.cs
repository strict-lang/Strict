namespace TestPackage;

public class ReduceButGrow
{
	private List<int> numbers = new List<int>();
	public int GetMultiplicationOfNumbers()
	{
		foreach (var index in numbers)
			value;
	}

	[Test]
	public void GetMultiplicationOfNumbersTest()
	{
		Assert.That(() => new ReduceButGrow(2, 3, 4, 5).GetMultiplicationOfNumbers() == 120));
		Assert.That(() => new ReduceButGrow(120, 5, 40, 0).GetMultiplicationOfNumbers() == 0));
		Assert.That(() => new ReduceButGrow(2, 2, 2, 2).GetMultiplicationOfNumbers() == 16));
	}
}