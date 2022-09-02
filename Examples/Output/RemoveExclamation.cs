namespace SourceGeneratorTests;

public class RemoveExclamation
{
	public string Remove(string input)
	{
		var result = "";
		for (var index = 0; index < input.Length; index++)
			if (input[index] != '!')
				result = result + input[index];
		return result;
	}
}