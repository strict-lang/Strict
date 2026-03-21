namespace TestPackage;

public class Fibonacci
{
	private int number;
	public int GetNthFibonacci()
	{
		var first = 1;
		var second = 1;
		foreach (var index in new Range(2, number))
				var next = first + second;
				first = second;
				second = next;
		second;
	}

	[Test]
	public void GetNthFibonacciTest()
	{
		Assert.That(() => new Fibonacci(5).GetNthFibonacci() == 5));
		Assert.That(() => new Fibonacci(10).GetNthFibonacci() == 55));
	}
}