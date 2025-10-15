using Strict.Language.Tests;

namespace Strict.HighLevelRuntime.Tests;

public sealed class ExecutionContextTests
{
	[Test]
	public void SetAndGetVariable()
	{
		var pkg = TestPackage.Instance;
		var ctx = new ExecutionContext();
		var num = pkg.FindType(Strict.Language.Base.Number)!;
		var val = new ValueInstance(num, 123);
		ctx.Set("answer", val);
		Assert.That(ctx.Get("answer"), Is.SameAs(val));
	}

	[Test]
	public void ParentLookupWorks()
	{
		var pkg = TestPackage.Instance;
		var parent = new ExecutionContext();
		var child = new ExecutionContext { Parent = parent };
		var num = pkg.FindType(Language.Base.Number)!;
		parent.Set("x", new ValueInstance(num, 5));
		Assert.That(child.Get("x").Value, Is.EqualTo(5));
	}

	[Test]
	public void GetUnknownVariableThrows() =>
		Assert.That(() => new ExecutionContext().Get("unknown"),
			Throws.TypeOf<ExecutionContext.VariableNotFound>());
}