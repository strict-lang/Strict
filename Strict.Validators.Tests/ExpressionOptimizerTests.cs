#if TODO
namespace Strict.Validators.Tests;

public sealed class ExpressionOptimizerTests
{
	[SetUp]
	public void Setup()
	{
		type = new Type(new TestPackage(), new TypeLines(nameof(ExpressionOptimizerTests), ""));
		parser = new MethodExpressionParser();
	}

	private Type type = null!;
	private ExpressionParser parser = null!;

	[Test]
	public void ConstantFolding_StringToNumber_Success()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\tconstant folded = \"5\" to Number",
			"\tfolded + 1"
		]);
		Assert.DoesNotThrow(() => new ExpressionOptimizer([method]).Validate());
	}

	[Test]
	public void ConstantFolding_ImpossibleCast_Fails()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\tconstant folded = \"abc\" to Number",
			"\tfolded + 1"
		]);
		Assert.Throws<ExpressionOptimizer.ImpossibleConstantCast>(() => new ExpressionOptimizer([method]).Validate());
	}

	[Test]
	public void SimplePropagation_ConstantPropagation_Success()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\tconstant a = 2",
			"\tconstant b = a + 3",
			"\tb * 2"
		]);
		Assert.DoesNotThrow(() => new ExpressionOptimizer([method]).Validate());
	}

	[Test]
	public void ValidatorIsStateless_DoesNotTouchRuntime()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\tconstant a = 1",
			"\ta + 1"
		]);
		var validator = new ExpressionOptimizer([method]);
		validator.Validate();
		// No runtime state should be changed, so running again should not throw
		Assert.DoesNotThrow(() => validator.Validate());
	}
}
#endif