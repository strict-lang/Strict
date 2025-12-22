namespace Strict.Expressions.Tests;

public sealed class BodyTests : TestExpressions
{
	[Test]
	public void FindVariableValue() =>
		Assert.That(
			new Body(method).AddVariable("num", new Number(method, 5), false).FindVariable("num")!.
				InitialValue, Is.EqualTo(new Number(method, 5)));

	[Test]
	public void FindParentVariableValue() =>
		Assert.That(
			new Body(method, 0, new Body(method).AddVariable("str", new Text(method, "Hello"), false)).
				AddVariable("num", new Number(method, 5), false).FindVariable("str")!.InitialValue,
			Is.EqualTo(new Text(method, "Hello")));

	[Test]
	public void CannotUseVariableFromLowerScope() =>
		Assert.That(() => ParseExpression("if bla is 5", "\tconstant abc = \"abc\"", "logger.Log(abc)"),
			Throws.InstanceOf<Body.IdentifierNotFound>().With.Message.StartWith("abc"));

	[Test]
	public void UnknownVariable() =>
		Assert.That(() => ParseExpression("if bla is 5", "\tlogger.Log(unknownVariable)"),
			Throws.InstanceOf<Body.IdentifierNotFound>().With.Message.StartWith("unknownVariable"));

	[Test]
	public void CannotAccessAnotherMethodVariable()
	{
		var program = new Type(new Package(nameof(CannotAccessAnotherMethodVariable)),
			new TypeLines(nameof(CannotAccessAnotherMethodVariable),
				// @formatter:off
				"has logger",
				"Run",
				"\tconstant number = 5",
				"Add",
				"\tlogger.Log(number)")).ParseMembersAndMethods(new MethodExpressionParser());
		// @formatter:on
		Assert.That(
			() => program.Methods[1].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Body.IdentifierNotFound>().With.Message.StartWith("number"));
	}

	[Test]
	public void IsConstant()
	{
		var program = new Type(new Package(nameof(IsConstant)),
			new TypeLines(nameof(IsConstant),
				// @formatter:off
				"has logger",
				"Run",
				"\tconstant number = 5",
				"\tlogger.Log(number + number)")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(program.Methods[0].GetBodyAndParseIfNeeded().IsConstant, Is.False);
	}

	[Test]
	public void IfHasDifferentScopeThanMethod() =>
		Assert.That(ParseExpression("if bla is 5", "\tconstant abc = \"abc\"", "\tlogger.Log(abc)"),
			Is.EqualTo(new If(GetCondition(), CreateThenBlock())));

	private Expression CreateThenBlock()
	{
		var body = new Body(method);
		var expressions = new Expression[2];
		expressions[0] = new Declaration(body, "abc", new Text(method, "abc"));
		var arguments = new Expression[] { new VariableCall(body.FindVariable("abc")!) };
		expressions[1] = new MethodCall(member.Type.GetMethod("Log", arguments),
			new MemberCall(null, member), arguments);
		body.SetExpressions(expressions);
		return body;
	}

	[Test]
	public void IfAndElseHaveTheirOwnScopes() =>
		Assert.That(() => ParseExpression(
				// @formatter:off
				"if bla is 5",
				"\tconstant ifText = \"in if\"",
				"\tlogger.Log(ifText)",
				"else",
				"\tlogger.Log(ifText)"),
				// @formatter:on
			Throws.InstanceOf<Body.IdentifierNotFound>().With.Message.StartWith("ifText"));

	[Test]
	public void MissingThenDueToIncorrectChildBodyStart() =>
		Assert.That(() => ParseExpression(
				"if bla is 5",
				"constant abc = \"abc\"",
				"\tlogger.Log(abc)"),
			Throws.InstanceOf<If.MissingThen>());

	[Test]
	public void EmptyInputIsNotAllowed() =>
		Assert.That(() => new Body(method).Parse(),
			Throws.InstanceOf<SpanExtensions.EmptyInputIsNotAllowed>());

	[Test]
	public void CheckVariableCallCurrentValue()
	{
		var ifExpression = ParseExpression(
			"if bla is 5",
			"\tconstant abc = \"abc\"",
			"\tlogger.Log(abc)") as If;
		var variableCall =
			((ifExpression?.Then as Body)?.Expressions[1] as MethodCall)?.Arguments[0] as VariableCall;
		Assert.That(variableCall?.Variable.InitialValue.ToString(), Is.EqualTo("\"abc\""));
	}

	[Test]
	public void DuplicateVariableNameFound() =>
		Assert.That(() => ParseExpression("if bla is 5", "\tconstant abc = 5", "\tconstant abc = 5"),
			Throws.InstanceOf<Body.VariableNameIsAlreadyInUse>().With.Message.StartsWith("Variable abc"));

	[Test]
	public void DuplicateVariableInLowerScopeIsNotAllowed() =>
		Assert.That(
			() => ParseExpression("if bla is 5", "\tconstant outerScope = \"abc\"", "\tif bla is 5.0",
				"\t\tconstant outerScope = 5"),
			Throws.InstanceOf<Body.VariableNameIsAlreadyInUse>().With.Message.
				StartsWith("Variable outerScope"));

	[Test]
	public void ChildBodyReturnsFromThreeTabsToOneDirectly()
	{
		var program = new Type(new Package(nameof(ChildBodyReturnsFromThreeTabsToOneDirectly)),
			new TypeLines(nameof(ChildBodyReturnsFromThreeTabsToOneDirectly),
      // @formatter:off
      "has logger",
      "Run",
      "\tconstant number = 5",
      "\tfor Range(1, number)",
      "\t\tif index is number",
      "\t\t\tlet current = index",
      "\t\t\treturn current",
      "\tnumber")).ParseMembersAndMethods(new MethodExpressionParser());
		// @formatter:on
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.Tabs, Is.EqualTo(1));
		Assert.That(body.children[0].Tabs, Is.EqualTo(2));
		Assert.That(body.children[0].children[0].Tabs, Is.EqualTo(3));
		Assert.That(body.LineRange, Is.EqualTo(1..7));
		Assert.That(body.children[0].LineRange, Is.EqualTo(3..6));
		Assert.That(body.children[0].children[0].LineRange, Is.EqualTo(4..6));
	}

	[Test]
	public void CannotUpdateNonMutableVariable() =>
		Assert.That(
			() => new Variable("yo", false, number, new Body(method)).CheckIfWeCouldUpdateValue(number),
			Throws.InstanceOf<Body.ValueIsNotMutableAndCannotBeChanged>());

	[Test]
	public void CannotUpdateNumberToList() =>
		Assert.That(
			() => new Variable("yo", true, number, new Body(method)).CheckIfWeCouldUpdateValue(list),
			Throws.InstanceOf<Variable.NewExpressionDoesNotMatchVariableType>());
}