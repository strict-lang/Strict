namespace Strict.Expressions.Tests;

public sealed class IfAdvancedTests : TestExpressions
{
	[Test]
	public void ParseIf() =>
		Assert.That(ParseExpression("if five is 5", "\tlogger.Log(\"Hey\")"),
			Is.EqualTo(new If(GetCondition(), GetThen())));

	private MethodCall GetThen() =>
		new(member.Type.Methods[0], new MemberCall(null, member), [new Text(type, "Hey")]);

	[Test]
	public void ParseIfNot() =>
		Assert.That(ParseExpression("if five is not 5", "\tlogger.Log(\"Hey\")"),
			Is.EqualTo(new If(GetCondition(true), GetThen())));

	[Test]
	public void ParseIfElse() =>
		Assert.That(ParseExpression("if five is 5", "\tlogger.Log(\"Hey\")", "else", "\tRun"),
			Is.EqualTo(new If(GetCondition(), GetThen(), 0, new MethodCall(method))).And.Not.
				EqualTo(new If(GetCondition(), GetThen())));

	[Test]
	public void ParseSelectorIf()
	{
		var program = new Type(Package,
			new TypeLines(nameof(ParseSelectorIf),
				// @formatter:off
				"has operation Text",
				"Run Number",
				"\tif operation is",
				"\t\t\"add\" then 1",
				"\t\t\"subtract\" then 2",
				"\t\t\"multiply\" then 3",
				"\t\t\"divide\" then 4")).ParseMembersAndMethods(new MethodExpressionParser());
		// @formatter:on
		var expression = program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(expression.ToString(), Is.EqualTo(string.Join(Environment.NewLine,
			"if operation is",
			"\t\"add\" then 1",
			"\t\"subtract\" then 2",
			"\t\"multiply\" then 3",
			"\t\"divide\" then 4")));
	}

	[Test]
	public void ParseSelectorIfWithElse()
	{
		var program = new Type(Package,
			new TypeLines(nameof(ParseSelectorIfWithElse),
				// @formatter:off
				"has operation Text",
				"Run Number",
				"\tif operation is",
				"\t\t\"add\" then 1",
				"\t\t\"subtract\" then 2",
				"\t\telse 3")).ParseMembersAndMethods(new MethodExpressionParser());
		// @formatter:on
		var expression = program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(expression.ToString(), Is.EqualTo(string.Join(Environment.NewLine,
			"if operation is",
			"\t\"add\" then 1",
			"\t\"subtract\" then 2",
			"\telse 3")));
	}

	[TestCase("constant result = true then true else false")]
	[TestCase("constant result = false then \"Yes\" else \"No\"")]
	[TestCase("constant result = 5 is 5 then (1, 2) else (3, 4)")]
	[TestCase("constant result = 5 + (false then 1 else 2)")]
	[TestCase("constant result = 5 is not 4 then (1, 2) else (3, 4)")]
	public void ValidConditionalExpressions(string code)
	{
		var body = (Body)ParseExpression(code, "result is Number");
		var assignment = (Declaration)body.Expressions[0];
		Assert.That(assignment.Value, Is.InstanceOf<If>().Or.InstanceOf<Binary>());
	}

	[Test]
	public void ConditionalExpressionsCannotBeNested() =>
		Assert.That(() => ParseExpression("constant result = true then true else (5 is 5 then false else true)"),
			Throws.InstanceOf<If.ConditionalExpressionsCannotBeNested>());

	[TestCase("logger.Log(true then \"Yes\" else \"No\")")]
	[TestCase("logger.Log(true then \"Yes\" + \"text\" else \"No\")")]
	[TestCase("logger.Log(\"Result\" + (true then \"Yes\" else \"No\"))")]
	[TestCase("logger.Log((true then \"Yes\" else \"No\") + \"Result\")")]
	[TestCase("5 is 5 then false else true")]
	[TestCase("6 is 5 then true else false")]
	public void ConditionalExpressionsAsPartOfOtherExpression(string code) =>
		Assert.That(ParseExpression(code).ToString(), Is.EqualTo(code));

	[Test]
	public void ReturnTypeOfThenMustMatchMethodReturnType()
	{
		var program = new Type(Package,
			new TypeLines(nameof(ReturnTypeOfThenMustMatchMethodReturnType),
				"has logger",
				"InvalidRun Number",
				"	if 5 is 5",
				"		constant file = File(\"test.txt\")",
				"		return \"5\"")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Body.ChildBodyReturnTypeMustMatchMethod>().With.Message.Contains(
				"Last expression return \"5\" return type: TestPackage/Text is not matching with " +
				"expected method return type: TestPackage/Number in method line: 3"));
	}

	private static readonly Package Package = new(nameof(IfTests));

	[Test]
	public void ReturnTypeOfElseMustMatchMethodReturnType()
	{
		var program = new Type(Package, new TypeLines(
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
				"Last expression return \"Hello\" return type: TestPackage/Text is not matching with " +
				"expected method return type: TestPackage/Number in method line: 4"));
	}

	[Test]
	public void ThenReturnsImplementedTypeOfMethodReturnType()
	{
		var program = new Type(Package,
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
		var program = new Type(Package,
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
		Assert.That(ParseExpression("if five is 5", "\tlogger.Log(\"Hey\")", "else if five is 5", "\tlogger.Log(\"Hey\")"),
			Is.EqualTo(new If(GetCondition(), GetThen(), 0, new If(GetCondition(), GetThen()))));

	[TestCase("else if five is 6")]
	[TestCase("else if")]
	[TestCase("if five is 5", "\tlogger.Log(\"Hey\")", "else if")]
	public void UnexpectedElseIf(params string[] code) =>
		Assert.That(() => ParseExpression(code),
			Throws.InstanceOf<If.UnexpectedElse>());

	[Test]
	public void ElseIfWithoutThen() =>
		Assert.That(() => ParseExpression("if five is 5", "\tlogger.Log(\"Hey\")", "else if five is 5"),
			Throws.InstanceOf<If.MissingThen>());

	[Test]
	public void ValidMultipleElseIf()
	{
		var program = new Type(Package,
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
		var program = new Type(Package,
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
		var program = new Type(Package, new TypeLines(
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
		Assert.That(ParseExpression("if five is not in (5)", "\tlogger.Log(\"Hey\")"),
			Is.EqualTo(new If(CreateNot(CreateBinary(list, BinaryOperator.In, new MemberCall(null, five))),
				GetThen())));

	[Test]
	public void ParseIsIn() =>
		Assert.That(ParseExpression("if five is in (5)", "\tlogger.Log(\"Hey\")"),
			Is.EqualTo(new If(CreateBinary(list, BinaryOperator.In, new MemberCall(null, five)),
				GetThen())));
}