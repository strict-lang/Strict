using Strict.Language;
using Strict.Expressions;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.Validators.Tests;

public sealed class ConstantFoldingAndPropagationTests
{
	[SetUp]
	public void Setup()
	{
		type = new Type(new TestPackage(), new TypeLines(nameof(ConstantFoldingAndPropagationTests), ""));
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
		Assert.DoesNotThrow(() => new ConstantFoldingValidator([method]).Validate());
	}

	[Test]
	public void ConstantFolding_ImpossibleCast_Fails()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\tconstant folded = \"abc\" to Number",
			"\tfolded + 1"
		]);
		Assert.Throws<ConstantFoldingValidator.ImpossibleConstantCast>(() => new ConstantFoldingValidator([method]).Validate());
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
		Assert.DoesNotThrow(() => new ConstantFoldingValidator([method]).Validate());
	}

	[Test]
	public void ValidatorIsStateless_DoesNotTouchRuntime()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\tconstant a = 1",
			"\ta + 1"
		]);
		var validator = new ConstantFoldingValidator([method]);
		validator.Validate();
		// No runtime state should be changed, so running again should not throw
		Assert.DoesNotThrow(() => validator.Validate());
	}
}
