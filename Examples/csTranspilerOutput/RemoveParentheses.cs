namespace TestPackage;

public class RemoveParentheses
{
	private string text = new string();
	public string Remove()
	{
		var parentheses = 0;
		var result = "";
		foreach (var index in text)
			if (value == "(")
				parentheses = parentheses + 1;
			else
				if (value == ")")
					parentheses = parentheses - 1;
				else
					if (parentheses == 0)
						result = result + value;
		result;
	}

	[Test]
	public void RemoveTest()
	{
		Assert.That(() => new RemoveParentheses("example(unwanted thing)example").Remove() == "exampleexample"));
	}
}