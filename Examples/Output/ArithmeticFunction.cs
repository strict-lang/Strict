namespace SourceGeneratorTests;

public class ArithmeticFunction
{
	private int first;
	private int second;
	private Text operation;
	public int Calculate()
	{
		if (operation == "add")
			return first + second;
		if (operation == "subtract")
			return first - second;
		if (operation == "multiply")
			return first * second;
		if (operation == "divide")
			return first / second;
	}
}