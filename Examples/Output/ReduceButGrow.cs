namespace SourceGeneratorTests;

public class ReduceButGrow
{
	public double GetMultiplicationOfNumbers(double[] numbers)
	{
		var result = 1.0;
		for (var index = 0; index < numbers.Length; index++)
			result = result * numbers[index];
		return result;
	}
}