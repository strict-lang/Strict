using Strict.Expressions;

namespace Strict.Optimizers.Tests;

public class TestOptimizers
{
	[SetUp]
	public void GetTypes()
	{
		var package = TestPackage.Instance;
		booleanType = TestPackage.Instance.GetType(Type.Boolean);
		numberType = TestPackage.Instance.GetType(Type.Number);
	}

	protected Type numberType = null!;
	protected Type booleanType = null!;
	public ValueInstance Num(double value) => new(numberType, value);
}