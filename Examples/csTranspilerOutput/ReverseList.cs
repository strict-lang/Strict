namespace TestPackage;

public class ReverseList
{
	private List<int> numbers = new List<int>();
	public List<int> Reverse()
	{
		foreach (var index in new Range(0, numbers.Length()).Reverse())
			numbers[index];
	}

	[Test]
	public void ReverseTest()
	{
		Assert.That(() => new ReverseList(1, 2, 3, 4).Reverse() == (4, 3, 2, 1)));
	}
}