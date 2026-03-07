namespace Strict.Runtime.Tests;

public sealed class CallFrameTests
{
	private static readonly ValueInstance SomeNumber =
		new(TestPackage.Instance.GetType(Type.Number), 42.0);

	[Test]
	public void TryGetReturnsFalseWhenVariableNotFoundInRootFrame()
	{
		var frame = new CallFrame();
		Assert.That(frame.TryGet("missing", out _), Is.False);
	}

	[Test]
	public void TryGetReturnsFalseForParentLocalVariableNotExposedAsMember()
	{
		var parent = new CallFrame();
		parent.Set("local", SomeNumber);
		var child = new CallFrame(parent);
		Assert.That(child.TryGet("local", out _), Is.False);
	}

	[Test]
	public void TryGetFindsMemberVariableFromParent()
	{
		var parent = new CallFrame();
		parent.Set("count", SomeNumber, isMember: true);
		var child = new CallFrame(parent);
		Assert.That(child.TryGet("count", out _), Is.True);
	}

	[Test]
	public void ClearRemovesLocalVariables()
	{
		var frame = new CallFrame();
		frame.Set("x", SomeNumber);
		frame.Clear();
		Assert.That(frame.TryGet("x", out _), Is.False);
	}

	[Test]
	public void ClearRemovesMemberVariablesFromChildVisibility()
	{
		var parent = new CallFrame();
		parent.Set("count", SomeNumber, isMember: true);
		parent.Clear();
		var child = new CallFrame(parent);
		Assert.That(child.TryGet("count", out _), Is.False);
	}
}
