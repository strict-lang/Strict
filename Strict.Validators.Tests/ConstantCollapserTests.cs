namespace Strict.Validators.Tests;

public sealed class ConstantCollapserTests
{
	[SetUp]
	public void Setup()
	{
		type = new Type(new TestPackage(),
			new TypeLines(nameof(ConstantCollapserTests), "has logger", "Run", "\tlogger.Log(5)"));
		parser = new MethodExpressionParser();
		type.ParseMembersAndMethods(parser);
		collapser = new ConstantCollapser();
	}

	private Type type = null!;
	private ExpressionParser parser = null!;
	private ConstantCollapser collapser = null!;

	[Test]
	public void FoldStringToNumberToJustNumber()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\tconstant folded = \"5\" to Number",
			"\tfolded + 1"
		]);
		Assert.That(() => collapser.Visit(method), Throws.Nothing);
		new ConstantCollapser().Visit(method, true);
		new ConstantUsagesOptimizer().Visit(method, true);
		var methodExpression = method.GetBodyAndParseIfNeeded();
		Assert.That(methodExpression, Is.InstanceOf<Number>());
		Assert.That(((Number)methodExpression).Data, Is.EqualTo(5));
	}

	/*TODO
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
		Assert.That(() => new ExpressionOptimizer([method]).Validate(), Throws.Nothing);
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
		Assert.That(() => validator.Validate(), Throws.Nothing);
	}
	*/
}