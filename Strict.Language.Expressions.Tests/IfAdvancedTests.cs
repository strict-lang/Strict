using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class IfAdvancedTests : TestExpressions
{
	[Test]
	public void ParseIf() =>
		Assert.That(ParseExpression("if bla is 5", "\tlog.Write(\"Hey\")"),
			Is.EqualTo(new If(GetCondition(), GetThen())));

	private MethodCall GetThen() =>
		new(member.Type.Methods[0], new MemberCall(null, member), [new Text(type, "Hey")]);

	[Test]
	public void ParseIfNot() =>
		Assert.That(ParseExpression("if bla is not 5", "\tlog.Write(\"Hey\")"),
			Is.EqualTo(new If(GetCondition(true), GetThen())));

	[Test]
	public void ParseIfElse() =>
		Assert.That(ParseExpression("if bla is 5", "\tlog.Write(\"Hey\")", "else", "\tRun"),
			Is.EqualTo(new If(GetCondition(), GetThen(), new MethodCall(method))).And.Not.
				EqualTo(new If(GetCondition(), GetThen())));

	[TestCase("constant result = true ? true else false")]
	[TestCase("constant result = false ? \"Yes\" else \"No\"")]
	[TestCase("constant result = 5 is 5 ? (1, 2) else (3, 4)")]
	[TestCase("constant result = 5 + (false ? 1 else 2)")]
	[TestCase("constant result = 5 is not 4 ? (1, 2) else (3, 4)")]
	public void ValidConditionalExpressions(string code)
	{
		var expression = ParseExpression(code);
		Assert.That(expression, Is.InstanceOf<ConstantDeclaration>());
		var assignment = expression as ConstantDeclaration;
		Assert.That(assignment?.Value, Is.InstanceOf<If>().Or.InstanceOf<Binary>());
	}

	[Test]
	public void ConditionalExpressionsCannotBeNested() =>
		Assert.That(() => ParseExpression("constant result = true ? true else (5 is 5 ? false else true)"),
			Throws.InstanceOf<If.ConditionalExpressionsCannotBeNested>());

	[TestCase("log.Write(true ? \"Yes\" else \"No\")")]
	[TestCase("log.Write(true ? \"Yes\" + \"text\" else \"No\")")]
	[TestCase("log.Write(\"Result\" + (true ? \"Yes\" else \"No\"))")]
	[TestCase("log.Write((true ? \"Yes\" else \"No\") + \"Result\")")]
	[TestCase("constant something = 5 is 5 ? false else true")]
	[TestCase("6 is 5 ? true else false")]
	public void ConditionalExpressionsAsPartOfOtherExpression(string code) =>
		Assert.That(ParseExpression(code).ToString(), Is.EqualTo(code));

	[Test]
	public void ReturnTypeOfThenMustMatchMethodReturnType()
	{
		var program = new Type(new Package(nameof(IfTests)),
			new TypeLines(nameof(ReturnTypeOfThenMustMatchMethodReturnType),
				"has log",
				"InvalidRun Number",
				"	if 5 is 5",
				"		constant file = File(\"test.txt\")",
				"		return \"5\"")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(), Throws.InstanceOf<Body.ChildBodyReturnTypeMustMatchMethod>().With.Message.Contains("Child body return type: TestPackage.Text is not matching with Parent return type: TestPackage.Number in method line: 3"));
	}

	[Test]
	public void ReturnTypeOfElseMustMatchMethodReturnType()
	{
		var program = new Type(new Package(nameof(IfTests)), new TypeLines(
			nameof(ReturnTypeOfElseMustMatchMethodReturnType),
			// @formatter:off
			"has log",
			"InvalidRun Text",
			"	if 5 is 5",
			"		constant file = File(\"test.txt\")",
			"		return \"Hello\"",
			"	else",
			"		return true")).ParseMembersAndMethods(new MethodExpressionParser());
		// @formatter:on
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(), Throws.InstanceOf<Body.ChildBodyReturnTypeMustMatchMethod>().With.Message.Contains("Child body return type: TestPackage.Boolean is not matching with Parent return type: TestPackage.Text in method line: 5"));
	}

	[Test]
	public void ThenReturnsImplementedTypeOfMethodReturnType()
	{
		var program = new Type(new Package(nameof(IfTests)),
			new TypeLines(nameof(ThenReturnsImplementedTypeOfMethodReturnType),
				// @formatter:off
				"has log",
				"InvalidRun Number",
				"	InvalidRun is 6",
				"	if 5 is 5",
				"		constant file = File(\"test.txt\")",
				"		return Character(5)",
				"	6")).ParseMembersAndMethods(new MethodExpressionParser());
		// @formatter:on
		Assert.That(((Body)program.Methods[0].GetBodyAndParseIfNeeded()).children[0].ReturnType.ToString(), Is.EqualTo("TestPackage.Number"));
	}

	[Test]
	public void MultiLineThenAndElseWithMatchingMethodReturnType()
	{
		var program = new Type(new Package(nameof(IfTests)),
			new TypeLines(nameof(MultiLineThenAndElseWithMatchingMethodReturnType),
				// @formatter:off
				"has log",
				"Run Text",
				"	if 5 is 5",
				"		constant file = File(\"test.txt\")",
				"		return \"Hello\"",
				"	else",
				"		return \"Hi\"",
				"	\"don't matter\"")).ParseMembersAndMethods(new MethodExpressionParser());
		// @formatter:on
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.ReturnType.ToString(), Is.EqualTo("TestPackage.Text"));
		Assert.That(body.children[0].ReturnType.ToString(), Is.EqualTo("TestPackage.Text"));
		Assert.That(body.children[1].ReturnType.ToString(), Is.EqualTo("TestPackage.Text"));
	}

	[Test]
	public void ParseElseIf() =>
		Assert.That(ParseExpression("if bla is 5", "\tlog.Write(\"Hey\")", "else if bla is 5", "\tlog.Write(\"Hey\")"),
			Is.EqualTo(new If(GetCondition(), GetThen(), new If(GetCondition(), GetThen()))));

	[TestCase("else if bla is 6")]
	[TestCase("else if")]
	[TestCase("if bla is 5", "\tlog.Write(\"Hey\")", "else if")]
	public void UnexpectedElseIf(params string[] code) =>
		Assert.That(() => ParseExpression(code),
			Throws.InstanceOf<If.UnexpectedElse>());

	[Test]
	public void ElseIfWithoutThen() =>
		Assert.That(() => ParseExpression("if bla is 5", "\tlog.Write(\"Hey\")", "else if bla is 5"),
			Throws.InstanceOf<If.MissingThen>());

	[Test]
	public void ValidMultipleElseIf()
	{
		var program = new Type(new Package(nameof(IfTests)),
			new TypeLines(nameof(ValidMultipleElseIf),
				// @formatter:off
				"has log",
				"Run Text",
				"	if 5 is 5",
				"		return \"Hello\"",
				"	else if 6 is 6",
				"		log.Write(\"Hi\")",
				"		return \"Hi\"",
				"	else if 7 is 7",
				"		log.Write(\"Hello\")",
				"		return \"Hello\"",
				"	\"don't matter\"")).ParseMembersAndMethods(new MethodExpressionParser());
		// @formatter:on
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.ToString(), Is.EqualTo(string.Join(Environment.NewLine,
			"if 5 is 5",
			"	return \"Hello\"",
			"else if 6 is 6",
			"	log.Write(\"Hi\")",
			"	return \"Hi\"",
			"else if 7 is 7",
			"	log.Write(\"Hello\")",
			"	return \"Hello\"",
			"\"don't matter\"")), body.ToString());
		Assert.That(body.children[1].ReturnType.ToString(), Is.EqualTo("TestPackage.Text"));
		Assert.That(body.children.Count, Is.EqualTo(3));
	}

	[Test]
	public void ElseIfMissingThen()
	{
		var program = new Type(new Package(nameof(IfTests)),
			new TypeLines(nameof(ElseIfMissingThen),
				// @formatter:off
				"has log",
				"ValidRun Text",
				"	if 5 is 5",
				"		constant file = File(\"test.txt\")",
				"		return \"Hello\"",
				"	else if 6 is 6",
				"	constant something = \"Hi\"",
				"	\"don't matter\"")).ParseMembersAndMethods(new MethodExpressionParser());
		// @formatter:on
		Assert.That(() => (Body)program.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<If.MissingThen>());
	}

	[Test]
	public void MultiLineElseWithMismatchingReturnType()
	{
		var program = new Type(new Package(nameof(IfTests)), new TypeLines(
			nameof(MultiLineElseWithMismatchingReturnType),
			// @formatter:off
			"has log",
			"MismatchingElseIfReturn Text",
			"	if 5 is 5",
			"		constant file = File(\"test.txt\")",
			"		return \"Hello\"",
			"	else if 6 is 6",
			"		return true",
			"	\"don't matter\"")).ParseMembersAndMethods(new MethodExpressionParser());
		// @formatter:on
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Body.ChildBodyReturnTypeMustMatchMethod>());
	}

	[Test]
	public void ParseIsNotIn() =>
		Assert.That(ParseExpression("if bla is not in (5)", "\tlog.Write(\"Hey\")"),
			Is.EqualTo(new If(CreateBinary(new MemberCall(null, bla), BinaryOperator.IsNotIn, list),
				GetThen())));

	[Test]
	public void ParseIsIn() =>
		Assert.That(ParseExpression("if bla is in (5)", "\tlog.Write(\"Hey\")"),
			Is.EqualTo(new If(CreateBinary(new MemberCall(null, bla), BinaryOperator.IsIn, list),
				GetThen())));
}