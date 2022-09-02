namespace SourceGeneratorTests;

public class Fibonacci
{
	public int GetNthFibonacci(double n)
	{
		var first = 1;
		var second = 1;
		var next = 1;
		for(var index = 2; index < n; index++)
		{
			next = first + second;
			first = second;
			second = next;
		}
		return next;
	}
}
