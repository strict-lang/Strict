using System.Reflection.Metadata;
using System.Text;
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
		Assert.That(((Number)method.GetBodyAndParseIfNeeded()).Data.Number, Is.EqualTo(6));
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
		Assert.That(((Text)method.GetBodyAndParseIfNeeded()).Data.Text, Is.EqualTo("5yo"));
	}

	[Test]
	public void FoldTwoConstants()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\tconstant hi = \"Hi\"",
			"\thi + hi"
		]);
		collapser.Visit(method, true);
		Assert.That(((Text)method.GetBodyAndParseIfNeeded()).Data.Text, Is.EqualTo("HiHi"));
	}

	[Test]
	public void FoldTwoMembers()
	{
		using var foldType = new Type(TestPackage.Instance,
			new TypeLines(nameof(FoldTwoMembers),
				"has one = 1",
				"has two = 2",
				"Run Number",
				"\tone + two"));
		foldType.ParseMembersAndMethods(parser);
		collapser.Visit(foldType.Methods[0], true);
		Assert.That(((Number)foldType.Methods[0].GetBodyAndParseIfNeeded()).Data.Number, Is.EqualTo(3));
	}

	[Test]
	public void FoldBooleans()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\ttrue or false and true"
		]);
		collapser.Visit(method, true);
		Assert.That(((Boolean)method.GetBodyAndParseIfNeeded()).Data.Boolean, Is.EqualTo(true));
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
		Assert.That(((Number)method.GetBodyAndParseIfNeeded()).Data.Number, Is.EqualTo(10));
	}

	[Test]
	public void FoldMemberInitialValueExpressions()
	{
		using var simpleType = new Type(TestPackage.Instance,
			new TypeLines(nameof(FoldMemberInitialValueExpressions),
				"constant number = 17 + 4", "Run", "\tnumber"));
		simpleType.ParseMembersAndMethods(parser);
		collapser.Visit(simpleType, true);
		Assert.That(((Number)simpleType.Members[0].InitialValue!).Data.Number, Is.EqualTo(21));
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
		Assert.That(((Number)simpleType.Methods[0].Parameters[1].DefaultValue!).Data.Number,
			Is.EqualTo(21));
		Assert.That(((Number)simpleType.Methods[^1].GetBodyAndParseIfNeeded()).Data.Number,
			Is.EqualTo(3));
	}

	[Test]
	public void FoldNestedBinaryOnLeftSide()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\t(1 + 2) * 3"
		]);
		collapser.Visit(method, true);
		Assert.That(((Number)method.GetBodyAndParseIfNeeded()).Data.Number, Is.EqualTo(9));
	}

	[Test]
	public void FoldNestedBinaryOnRightSide()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\t1 * (2 + 3)"
		]);
		collapser.Visit(method, true);
		Assert.That(((Number)method.GetBodyAndParseIfNeeded()).Data.Number, Is.EqualTo(5));
	}

	[Test]
	public void FoldTextPlusNumber()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\t\"value\" + 5"
		]);
		collapser.Visit(method, true);
		Assert.That(((Text)method.GetBodyAndParseIfNeeded()).Data.Text, Is.EqualTo("value5"));
	}

	[Test]
	public void FoldMinusOperation()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\t9 - 4"
		]);
		collapser.Visit(method, true);
		Assert.That(((Number)method.GetBodyAndParseIfNeeded()).Data.Number, Is.EqualTo(5));
	}

	[Test]
	public void FoldDivideOperation()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\t9 / 2"
		]);
		collapser.Visit(method, true);
		Assert.That(((Number)method.GetBodyAndParseIfNeeded()).Data.Number, Is.EqualTo(4.5));
	}

	[Test]
	public void KeepExpressionWhenOperatorCannotBeCollapsed()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\ttrue xor false"
		]);
		collapser.Visit(method, true);
		Assert.That(method.GetBodyAndParseIfNeeded(), Is.InstanceOf<Binary>());
	}

	[Test]
	public void FoldTextPlusBoolean()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\t\"hello\" + true"
		]);
		collapser.Visit(method, true);
		Assert.That(((Text)method.GetBodyAndParseIfNeeded()).Data.Text, Is.EqualTo("helloTrue"));
	}

	[Test]
	public void FoldsMemberWithBinaryInitialValueOnLeftSide()
	{
		using var testType = new Type(TestPackage.Instance,
			new TypeLines(nameof(FoldsMemberWithBinaryInitialValueOnLeftSide),
				"constant one = 2 + 3",
				"Run Number",
				"\tone * 2"));
		testType.ParseMembersAndMethods(parser);
		collapser.Visit(testType.Methods[0], true);
		Assert.That(((Number)testType.Methods[0].GetBodyAndParseIfNeeded()).Data.Number, Is.EqualTo(10));
	}

	[Test]
	public void FoldsMemberWithBinaryInitialValueOnRightSide()
	{
		using var testType = new Type(TestPackage.Instance,
			new TypeLines(nameof(FoldsMemberWithBinaryInitialValueOnRightSide),
				"constant one = 2 + 3",
				"Run Number",
				"\t2 * one"));
		testType.ParseMembersAndMethods(parser);
		collapser.Visit(testType.Methods[0], true);
		Assert.That(((Number)testType.Methods[0].GetBodyAndParseIfNeeded()).Data.Number, Is.EqualTo(10));
	}

	[Test]
	public void SubstitutesConstantMemberCallInBinaryWhenOtherSideIsNonConstant()
	{
		using var testType = new Type(TestPackage.Instance,
			new TypeLines("PartialBinaryCollapse",
				"constant one = 1",
				"has number",
				"AddConstant Number",
				"\tone + number",
				"Run",
				"\tnumber"));
		testType.ParseMembersAndMethods(parser);
		collapser.Visit(testType.Methods[0], true);
		Assert.That(((Binary)testType.Methods[0].GetBodyAndParseIfNeeded()).Instance,
			Is.InstanceOf<Number>());
	}

	[Test]
	public void UsedConstantComplexExpressionShouldNotBeFolded()
	{
		using var testType = new Type(TestPackage.Instance,
			new TypeLines(nameof(UsedConstantComplexExpressionShouldNotBeFolded),
				"has logger",
				"Run",
				"\tconstant program = " + nameof(UsedConstantComplexExpressionShouldNotBeFolded),
				"\tlogger.Log(program)"));
		testType.ParseMembersAndMethods(parser);
		collapser.Visit(testType.Methods[0], true);
		var body = (Body)testType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.Expressions, Has.Count.EqualTo(2));
	}
}