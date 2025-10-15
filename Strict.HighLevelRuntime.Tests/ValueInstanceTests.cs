namespace Strict.HighLevelRuntime.Tests;

public sealed class ValueInstanceTests
{
	[Test]
	public void ToStringShowsTypeAndValue()
	{
		var numberType = Language.Tests.TestPackage.Instance.FindType(Language.Base.Number)!;
		var v = new ValueInstance(numberType, 42);
		Assert.That(v.ToString(), Is.EqualTo("Number:42"));
	}
}