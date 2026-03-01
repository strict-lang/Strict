namespace TestPackage;

public class RemoveExclamation
{
	private string text = new string();
	public string Remove()
	{
		foreach (var index in text)
			if (value is not "!")
				value = value + value;
	}

	[Test]
	public void RemoveTest()
	{
		Assert.That(() => new RemoveExclamation("Hello There!").Remove() == "Hello There"));
		Assert.That(() => new RemoveExclamation("Hi!!!").Remove() == "Hi"));
		Assert.That(() => new RemoveExclamation("Wow! Awesome! There!").Remove() == "Wow Awesome"));
	}
}