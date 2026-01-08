using Strict.Language;
using Strict.Language.Tests;

namespace Strict.HighLevelRuntime.Tests;

public sealed class ExecutionContextTests
{
	[Test]
	public void SetAndGetVariable()
	{
		var pkg = TestPackage.Instance;
		var num = pkg.FindType(Base.Number)!;
		var ctx = new ExecutionContext(num);
		var val = new ValueInstance(num, 123);
		ctx.Set("answer", val);
		Assert.That(ctx.Get("answer"), Is.SameAs(val));
	}

	[Test]
	public void ParentLookupWorks()
	{
		var pkg = TestPackage.Instance;
		var num = pkg.FindType(Base.Number)!;
		var parent = new ExecutionContext(num);
		var child = new ExecutionContext(num) { Parent = parent };
		parent.Set("x", new ValueInstance(num, 5));
		Assert.That(child.Get("x").Value, Is.EqualTo(5));
	}

	[Test]
	public void GetUnknownVariableThrows() =>
		Assert.That(
			() => new ExecutionContext(TestPackage.Instance.FindType(Base.Number)!).Get("unknown"),
			Throws.TypeOf<ExecutionContext.VariableNotFound>());
}