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
	public static void Main()
	{
		Console.WriteLine("Fibonacci(10) = " + new Fibonacci(10).GetNthFibonacci());
		Console.WriteLine("Fibonacci(5) = " + new Fibonacci(5).GetNthFibonacci());
	}

	[Test]
	public void GetNthFibonacciTest()
	{
		Assert.That(() => new Fibonacci(5).GetNthFibonacci() == 5));
		Assert.That(() => new Fibonacci(10).GetNthFibonacci() == 55));
	}
}
