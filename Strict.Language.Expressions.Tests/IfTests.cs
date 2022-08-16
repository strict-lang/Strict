using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class IfTests : TestExpressions
{
	[Test]
	public void MissingCondition() =>
		Assert.That(() => ParseExpression("if"),
			Throws.InstanceOf<If.MissingCondition>().With.Message.
				Contains(@"TestPackage\dummy.strict:line 2"));

	[Test]
	public void InvalidCondition() =>
		Assert.That(() => ParseExpression("if 5", "\treturn 0"),
			Throws.InstanceOf<If.InvalidCondition>());

	[Test]
	public void ReturnTypeOfThenAndElseMustNotBeAny() =>
		Assert.That(() => ParseExpression("if 5 is 6", "\treturn 8", "else", "\treturn \"hello\"").ReturnType,
			Throws.InstanceOf<If.ReturnTypeOfThenAndElseMustHaveMatchingType>());

	[Test]
	public void ReturnTypeOfThenAndElseIsNumberAndCountIsValid() =>
		Assert.That(ParseExpression("if bla is 5", "\treturn Count(0)", "else", "\treturn 5").ReturnType,
			Is.EqualTo(type.GetType(Base.Number)));

	[Test]
	public void ReturnTypeOfThenAndElseIsCountAndCharacterIsValid() =>
		Assert.That(ParseExpression("if bla is 5", "\treturn Count(0)", "else", "\treturn Character(5)").ReturnType,
			Is.EqualTo(type.GetType(Base.Number)));

	[Test]
	public void ParseInvalidSpaceAfterElseIsNotAllowed() =>
		Assert.That(() => ParseExpression("else "), Throws.InstanceOf<Type.ExtraWhitespacesFoundAtEndOfLine>());

	[Test]
	public void ParseJustElseIsNotAllowed() =>
		Assert.That(() => ParseExpression("else"),
			Throws.InstanceOf<If.UnexpectedElse>().With.Message.Contains(@"at Run in "));

	[Test]
	public void ParseIncompleteThen() =>
		Assert.That(() => ParseExpression("if bla is 5"), Throws.InstanceOf<If.MissingThen>());

	[Test]
	public void ParseWrongIndentation() =>
		Assert.That(() => ParseExpression("if bla is 5", "Run"),
			Throws.InstanceOf<Method.InvalidIndentation>());

	[Test]
	public void ParseIf() =>
		Assert.That(ParseExpression("if bla is 5", "\tlog.Write(\"Hey\")"),
			Is.EqualTo(new If(GetCondition(), GetThen())));

	[Test]
	public void ParseMissingElseExpression() =>
		Assert.That(() => ParseExpression("if bla is 5", "\tRun", "else"),
			Throws.InstanceOf<If.UnexpectedElse>().With.Message.
				Contains(@"TestPackage\dummy.strict:line 4"));

	[Test]
	public void ParseIfElse() =>
		Assert.That(ParseExpression("if bla is 5", "\tlog.Write(\"Hey\")", "else", "\tRun"),
			Is.EqualTo(new If(GetCondition(), GetThen(), new MethodCall(method))).And.Not.
				EqualTo(new If(GetCondition(), GetThen())));

	private MethodCall GetThen() =>
		new(member.Type.Methods[0], new MemberCall(null, member),
			new Expression[] { new Text(type, "Hey") });

	private Binary GetCondition() =>
		CreateBinary(new MemberCall(null, bla), BinaryOperator.Is, number);

	[Test]
	public void ReturnGetHashCode()
	{
		var ifExpression = (If)ParseExpression("if bla is 5", "\tRun");
		Assert.That(ifExpression.GetHashCode(),
			Is.EqualTo(ifExpression.Condition.GetHashCode() ^ ifExpression.Then.GetHashCode()));
	}

	[Test]
	public void MissingElseExpression() =>
		Assert.That(() => ParseExpression("let result = true ? true"),
			Throws.InstanceOf<If.MissingElseExpression>());

	[Test]
	public void InvalidConditionInConditionalExpression() =>
		Assert.That(() => ParseExpression("let result = 5 ? true"),
			Throws.InstanceOf<UnknownExpression>());

	[Test]
	public void ReturnTypeOfThenAndElseMustHaveMatchingType() =>
		Assert.That(() => ParseExpression("let result = true ? true else 5"),
			Throws.InstanceOf<If.ReturnTypeOfThenAndElseMustHaveMatchingType>());

	[TestCase("let result = true ? true else false")]
	[TestCase("let result = false ? \"Yes\" else \"No\"")]
	[TestCase("let result = 5 is 5 ? (1, 2) else (3, 4)")]
	[TestCase("let result = 5 + (false ? 1 else 2)")]
	public void ValidConditionalExpressions(string code)
	{
		var expression = ParseExpression(code);
		Assert.That(expression, Is.InstanceOf<Assignment>()!);
		var assignment = expression as Assignment;
		Assert.That(assignment?.Value, Is.InstanceOf<If>().Or.InstanceOf<Binary>()!);
	}

	[Test]
	public void ConditionalExpressionsCannotBeNested() =>
		Assert.That(() => ParseExpression("let result = true ? true else (5 is 5 ? false else true)"),
			Throws.InstanceOf<If.ConditionalExpressionsCannotBeNested>());

	[TestCase("log.Write(true ? \"Yes\" else \"No\")")]
	[TestCase("log.Write(true ? \"Yes\" + \"text\" else \"No\")")]
	[TestCase("log.Write(\"Result\" + (true ? \"Yes\" else \"No\"))")]
	[TestCase("let something = 5 is 5 ? false else true")]
	[TestCase("6 is 5 ? true else false")]
	public void ConditionalExpressionsAsPartOfOtherExpression(string code) =>
		Assert.That(ParseExpression(code).ToString(), Is.EqualTo(code));

	[Test]
	public void IfHasDifferentScopeThanMethod()
	{
		Assert.That(() => ParseExpression(
				"if bla is 5",
				"\tlet abc = \"abc\"",
				"log.Write(abc)"),
			Throws.InstanceOf<IdentifierNotFound>());
		Assert.That(ParseExpression(
				"if bla is 5",
				"\tlet abc = \"abc\"",
				"\tlog.Write(abc)"),
			Is.EqualTo(new If(GetCondition(), CreateThenBlock())));
	}

	private Expression CreateThenBlock()
	{
		var expressions = new Expression[2];
		var body = new Body(method);//.GetType(Base.None), expressions);
		expressions[0] = new Assignment(body, "abc", new Text(method, "abc"));
		var arguments = new Expression[] { new VariableCall("abc", body.FindVariableValue("abc")!) };
		expressions[1] = new MethodCall(member.Type.GetMethod("Write", arguments), new MemberCall(null, member), arguments);
		body.SetAndValidateExpressions(expressions, new Method.Line[2]);
		return body.Expressions[0];
	}
}