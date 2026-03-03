using Strict.Expressions;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime.Tests;

public sealed class ExecutionContextTests
{
	[Test]
	public void SetAndGetVariable()
	{
		var numberType = TestPackage.Instance.GetType(Type.Number);
		var ctx = new ExecutionContext(numberType, numberType.Methods[0]);
		var val = new ValueInstance(numberType, 123);
		ctx.Set("answer", val);
		Assert.That(ctx.Get("answer", new Statistics()), Is.EqualTo(val));
	}

	[Test]
	public void ParentLookupWorks()
	{
		var numberType = TestPackage.Instance.GetType(Type.Number);
		var parent = new ExecutionContext(numberType, numberType.Methods[0]);
		var child = new ExecutionContext(numberType, numberType.Methods[0]) { Parent = parent };
		parent.Set("x", new ValueInstance(numberType, 5));
		Assert.That(child.Get("x", new Statistics()).Number, Is.EqualTo(5));
	}

	[Test]
	public void GetUnknownVariableThrows() =>
		Assert.That(
			() => new ExecutionContext(TestPackage.Instance.GetType(Type.Number),
				TestPackage.Instance.GetType(Type.Number).Methods[0]).Get("unknown", new Statistics()),
			Throws.TypeOf<ExecutionContext.VariableNotFound>());
}