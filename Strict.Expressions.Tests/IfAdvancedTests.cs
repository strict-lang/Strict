namespace Strict.Expressions.Tests;

public sealed class IfAdvancedTests : TestExpressions
{
	[Test]
	public void ParseIf() =>
		Assert.That(ParseExpression("if bla is 5", "\tlogger.Log(\"Hey\")"),
			Is.EqualTo(new If(GetCondition(), GetThen())));

	private MethodCall GetThen() =>
		new(member.Type.Methods[0], new MemberCall(null, member), [new Text(type, "Hey")]);

	[Test]
	public void ParseIfNot() =>
		Assert.That(ParseExpression("if bla is not 5", "\tlogger.Log(\"Hey\")"),
			Is.EqualTo(new If(GetCondition(true), GetThen())));

	[Test]
	public void ParseIfElse() =>
		Assert.That(ParseExpression("if bla is 5", "\tlogger.Log(\"Hey\")", "else", "\tRun"),
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

	[TestCase("logger.Log(true ? \"Yes\" else \"No\")")]
	[TestCase("logger.Log(true ? \"Yes\" + \"text\" else \"No\")")]
	[TestCase("logger.Log(\"Result\" + (true ? \"Yes\" else \"No\"))")]
	[TestCase("logger.Log((true ? \"Yes\" else \"No\") + \"Result\")")]
	[TestCase("constant something = 5 is 5 ? false else true")]
	[TestCase("6 is 5 ? true else false")]
	public void ConditionalExpressionsAsPartOfOtherExpression(string code) =>
		Assert.That(ParseExpression(code).ToString(), Is.EqualTo(code));

	[Test]
	public void ReturnTypeOfThenMustMatchMethodReturnType()
	{
		var program = new Type(new Package(nameof(IfTests)),
			new TypeLines(nameof(ReturnTypeOfThenMustMatchMethodReturnType),
				"has logger",
				"InvalidRun Number",
				"	if 5 is 5",
				"		constant file = File(\"test.txt\")",
				"		return \"5\"")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Body.ChildBodyReturnTypeMustMatchMethod>().With.Message.Contains(
				"Last expression return \"5\" return type: TestPackage.Text is not matching with expected " +
				"method return type: TestPackage.Number in method line: 3"));
	}

	[Test]
	public void ReturnTypeOfElseMustMatchMethodReturnType()
	{
		var program = new Type(new Package(nameof(IfTests)), new TypeLines(
			nameof(ReturnTypeOfElseMustMatchMethodReturnType),
			// @formatter:off
			"has logger",
			"InvalidRun Number",
			"	InvalidRun is Number",
			"	if 5 is 5",
			"		constant file = File(\"test.txt\")",
			"		return \"Hello\"",
			"	6")).ParseMembersAndMethods(new MethodExpressionParser());
		// @formatter:on
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Body.ChildBodyReturnTypeMustMatchMethod>().With.Message.Contains(
				"Last expression return \"Hello\" return type: TestPackage.Text is not matching with " +
				"expected method return type: TestPackage.Number in method line: 4"));
	}

	[Test]
	public void ThenReturnsImplementedTypeOfMethodReturnType()
	{
		var program = new Type(new Package(nameof(IfTests)),
			new TypeLines(nameof(ThenReturnsImplementedTypeOfMethodReturnType),
				// @formatter:off
				"has logger",
				"InvalidRun Number",
				"	InvalidRun is 6",
				"	if 5 is 5",
				"		constant file = File(\"test.txt\")",
				"		return Character(5)",
				"	6")).ParseMembersAndMethods(new MethodExpressionParser());
		// @formatter:on
		Assert.That(
			((Body)program.Methods[0].GetBodyAndParseIfNeeded()).children[0].ReturnType.ToString(),
			Is.EqualTo("TestPackage.Number"));
	}

	[Test]
	public void MultiLineThenAndElseWithMatchingMethodReturnType()
	{
		var program = new Type(new Package(nameof(IfTests)),
			new TypeLines(nameof(MultiLineThenAndElseWithMatchingMethodReturnType),
				// @formatter:off
				"has logger",
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
		Assert.That(ParseExpression("if bla is 5", "\tlogger.Log(\"Hey\")", "else if bla is 5", "\tlogger.Log(\"Hey\")"),
			Is.EqualTo(new If(GetCondition(), GetThen(), new If(GetCondition(), GetThen()))));

	[TestCase("else if bla is 6")]
	[TestCase("else if")]
	[TestCase("if bla is 5", "\tlogger.Log(\"Hey\")", "else if")]
	public void UnexpectedElseIf(params string[] code) =>
		Assert.That(() => ParseExpression(code),
			Throws.InstanceOf<If.UnexpectedElse>());

	[Test]
	public void ElseIfWithoutThen() =>
		Assert.That(() => ParseExpression("if bla is 5", "\tlogger.Log(\"Hey\")", "else if bla is 5"),
			Throws.InstanceOf<If.MissingThen>());

	[Test]
	public void ValidMultipleElseIf()
	{
		var program = new Type(new Package(nameof(IfTests)),
			new TypeLines(nameof(ValidMultipleElseIf),
				// @formatter:off
				"has logger",
				"Run Text",
				"	if 5 is 5",
				"		return \"Hello\"",
				"	else if 6 is 6",
				"		logger.Log(\"Hi\")",
				"		return \"Hi\"",
				"	else if 7 is 7",
				"		logger.Log(\"Hello\")",
				"		return \"Hello\"",
				"	\"don't matter\"")).ParseMembersAndMethods(new MethodExpressionParser());
		// @formatter:on
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.ToString(), Is.EqualTo(string.Join(Environment.NewLine,
			"if 5 is 5",
			"	return \"Hello\"",
			"else if 6 is 6",
			"	logger.Log(\"Hi\")",
			"	return \"Hi\"",
			"else if 7 is 7",
			"	logger.Log(\"Hello\")",
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
				"has logger",
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
			"has logger",
			"MismatchingElseIfReturn Text",
			"	if 5 is 5",
			"		constant file = File(\"test.txt\")",
			"		return \"Hello\"",
			"	else if 6 is 6",
			"		return true",
			"	\"don't matter\"")).ParseMembersAndMethods(new MethodExpressionParser());
		// @formatter:on
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<If.ReturnTypeOfThenAndElseMustHaveMatchingType>());
	}

	[Test]
	public void ParseIsNotIn() =>
		Assert.That(ParseExpression("if bla is not in (5)", "\tlogger.Log(\"Hey\")"),
			Is.EqualTo(new If(CreateBinary(new MemberCall(null, bla), BinaryOperator.IsNotIn, list),
				GetThen())));

	[Test]
	public void ParseIsIn() =>
		Assert.That(ParseExpression("if bla is in (5)", "\tlogger.Log(\"Hey\")"),
			Is.EqualTo(new If(CreateBinary(new MemberCall(null, bla), BinaryOperator.IsIn, list),
				GetThen())));
}