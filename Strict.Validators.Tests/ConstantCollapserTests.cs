using Boolean = Strict.Expressions.Boolean;

namespace Strict.Validators.Tests;

public sealed class ConstantCollapserTests
{
	[SetUp]
	public void Setup()
	{
		type = new Type(TestPackage.Instance,
			new TypeLines(nameof(ConstantCollapserTests), "has logger", "Run", "\tlogger.Log(5)"));
		parser = new MethodExpressionParser();
		type.ParseMembersAndMethods(parser);
		collapser = new ConstantCollapser();
	}

	private Type type = null!;
	private ExpressionParser parser = null!;
	private ConstantCollapser collapser = null!;

	[TearDown]
	public void TearDown() => type.Dispose();

	[Test]
	public void ComplainWhenAConstantIsUsedInANormalMember()
	{
		using var simpleType = new Type(TestPackage.Instance,
			new TypeLines(nameof(FoldMemberInitialValueExpressions),
				"has number = 17 + 4", "Run", "\tnumber"));
		simpleType.ParseMembersAndMethods(parser);
		// ReSharper disable once AccessToDisposedClosure
		Assert.That(() => collapser.Visit(simpleType, true),
			Throws.InstanceOf<ConstantCollapser.UseConstantHere>());
	}

	[Test]
	public void FoldTextToNumberToJustNumber()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\tconstant folded = \"5\" to Number",
			"\tfolded + 1"
		]);
		collapser.Visit(method, true);
		Assert.That(((Number)method.GetBodyAndParseIfNeeded()).Data, Is.EqualTo(6));
	}

	[Test]
	public void FoldNumberToTextToJustText()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\tconstant folded = 5 to Text",
			"\tfolded + \"yo\""
		]);
		collapser.Visit(method, true);
		Assert.That(((Text)method.GetBodyAndParseIfNeeded()).Data, Is.EqualTo("5yo"));
	}

	[Test]
	public void FoldBooleans()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\ttrue or false and true"
		]);
		collapser.Visit(method, true);
		Assert.That(((Boolean)method.GetBodyAndParseIfNeeded()).Data, Is.EqualTo(true));
	}

	[Test]
	public void ConstantFoldWithImpossibleCastFails()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\tconstant folded = \"abc\" to Number",
			"\tfolded + 1"
		]);
		Assert.Throws<FormatException>(() => collapser.Visit(method, true));
	}

	[Test]
	public void MultipleNestedConstantsGetFoldedToo()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\tconstant first = 2",
			"\tconstant second = first + 3",
			"\tsecond * 2"
		]);
		collapser.Visit(method, true);
		Assert.That(((Number)method.GetBodyAndParseIfNeeded()).Data, Is.EqualTo(10));
	}

	[Test]
	public void FoldMemberInitialValueExpressions()
	{
		using var simpleType = new Type(TestPackage.Instance,
			new TypeLines(nameof(FoldMemberInitialValueExpressions),
				"constant number = 17 + 4", "Run", "\tnumber"));
		simpleType.ParseMembersAndMethods(parser);
		collapser.Visit(simpleType, true);
		Assert.That(((Number)simpleType.Members[0].InitialValue!).Data, Is.EqualTo(21));
	}

	[Test]
	public void FoldParameterDefaultValueExpressions()
	{
		using var simpleType = new Type(TestPackage.Instance,
			new TypeLines(nameof(FoldParameterDefaultValueExpressions), "constant number = 1",
				"AddSomething(first Number, add = 17 + 4)", "\tfirst + add", "Run",
				"\tnumber + number * 2"));
		simpleType.ParseMembersAndMethods(parser);
		collapser.Visit(simpleType, true);
		Assert.That(((Number)(simpleType.Methods[0].Parameters[1].DefaultValue!)).Data,
			Is.EqualTo(21));
		Assert.That(((Number)simpleType.Methods[^1].GetBodyAndParseIfNeeded()).Data, Is.EqualTo(3));
	}
}