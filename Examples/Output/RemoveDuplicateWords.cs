namespace TestPackage;

public class RemoveDuplicateWords
{
	private List<string> texts = new List<string>();
	public List<string> Remove()
	{
		foreach (var value in texts)
			if (texts.Index(value) == index)
				value;
	}

	[Test]
	public void RemoveTest()
	{
		Assert.That(() => new RemoveDuplicateWords(("a", "b", "b")).Remove() == ("a", "b")));
		Assert.That(() => new RemoveDuplicateWords(("a", "b", "c")).Remove() == ("a", "b", "c")));
		Assert.That(() => new RemoveDuplicateWords(("hello", "hi", "hiiii", "hello")).Remove() == ("hello", "hi", "hiiii")));
	}
}