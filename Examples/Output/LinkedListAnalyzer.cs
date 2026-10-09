namespace TestPackage;

public class LinkedListAnalyzer
{
	private int maxCount;
	public int AnalyzeList(int steps)
	{
		var result = 0;
		foreach (var index in steps)
			if (result < maxCount)
				result = result + 1;
		result;
	}

	[Test]
	public void AnalyzeListTest()
	{
		Assert.That(() => new LinkedListAnalyzer(5).AnalyzeList(3) == 3));
		Assert.That(() => new LinkedListAnalyzer(2).AnalyzeList(3) == 2));
	}
}