using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime.Tests;

public sealed class ExecutionContextTests
{
	[SetUp]
	public void CreateType() => num = TestPackage.Instance.FindType(Base.Number)!;

	private Type num = null!;

	[Test]
	public void SetAndGetVariable()
	{
		var ctx = new ExecutionContext(num, num.Methods[0]);
		var val = ValueInstance.Create(num, 123);
		ctx.Set("answer", val);
		Assert.That(ctx.Get("answer", new Statistics()), Is.EqualTo(val));
	}

	[Test]
	public void ParentLookupWorks()
	{
		var parent = new ExecutionContext(num, num.Methods[0]);
		var child = new ExecutionContext(num, num.Methods[0]) { Parent = parent };
		parent.Set("x", ValueInstance.Create(num, 5));
		Assert.That(child.Get("x", new Statistics()).Value, Is.EqualTo(5));
	}

	[Test]
	public void GetUnknownVariableThrows() =>
		Assert.That(
			() => new ExecutionContext(num, num.Methods[0]).Get("unknown", new Statistics()),
			Throws.TypeOf<ExecutionContext.VariableNotFound>());
}