namespace SourceGeneratorTests;

public class ReduceButGrow
{
	public double GetMultiplicationOfNumbers(double[] numbers)
	{
		var result = 1D;
		for (var index = 0; index < numbers.Length; index++)
			result = result * numbers[index];
		return result;
	}
}