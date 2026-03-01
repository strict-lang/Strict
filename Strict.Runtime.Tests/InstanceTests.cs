namespace Strict.Runtime.Tests;

public sealed class InstanceTests : BaseVirtualMachineTests
{
	[Test]
	public void ListSubtractionRemovesExpressionInstance()
	{
		var targetExpression = new Value(NumberType, 5);
		var left = new Instance(ListType.GetGenericImplementation(NumberType),
			new List<Expression> { targetExpression, new Value(NumberType, 7) });
		var right = new Instance(Base.Number, targetExpression);
		var result = left - right;
		var resultElements = (List<Expression>)result.Value;
		Assert.That(resultElements, Has.Count.EqualTo(1));
		Assert.That(((Value)resultElements[0]).Data, Is.EqualTo(7));
	}
}