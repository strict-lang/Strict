namespace SourceGeneratorTests;

public class Beeramid
{
	public Beeramid(double bonus, double price)
	{
		this.bonus = bonus;
		this.price = price;
	}

	private double bonus;
	private double price;
	public int GetCompleteLevelCount() =>
		CalculateCompleteLevelCount(bonus / price, 0);

	private static int CalculateCompleteLevelCount(int numberOfCans, int levelCount)
	{
		var remainingCans = numberOfCans - (levelCount * levelCount);
		return remainingCans < ((levelCount + 1) * (levelCount + 1))
			? levelCount
			: CalculateCompleteLevelCount(remainingCans, levelCount + 1);
	}
}