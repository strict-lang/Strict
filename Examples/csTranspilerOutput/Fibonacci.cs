namespace TestPackage;

public class Fibonacci
{
	private int number;
	public int GetNthFibonacci()
	{
		var first = 1;
		var second = 1;
		var next = 1;
		foreach (var index in new Range(2, number))
		{
				next = first + second;
				first = second;
				second = next;
		}
		return next;
	}

	[Test]
	public void GetNthFibonacciTest()
	{
		Assert.That(() => new Fibonacci(5).GetNthFibonacci() == 5));
		Assert.That(() => new Fibonacci(10).GetNthFibonacci() == 55));
	}
}