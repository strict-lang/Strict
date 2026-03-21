namespace TestPackage;

public class RemoveParentheses
{
	private string text = new string();
	public string Remove()
	{
		var parentheses = 0;
		foreach (var index in text)
			if (value == "(")
				parentheses = parentheses.Increment;
			else
				if (value == ")")
					parentheses = parentheses.Decrement;
				else
					if (parentheses == 0)
						value;
	}

	[Test]
	public void RemoveTest()
	{
		Assert.That(() => new RemoveParentheses("example(unwanted thing)example").Remove() == "exampleexample"));
	}
}