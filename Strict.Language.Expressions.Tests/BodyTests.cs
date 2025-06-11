﻿using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public sealed class BodyTests : TestExpressions
{
	[Test]
	public void FindVariableValue() =>
		Assert.That(
			new Body(method).AddVariable("num", new Number(method, 5)).FindVariableValue("num"),
			Is.EqualTo(new Number(method, 5)));

	[Test]
	public void FindParentVariableValue() =>
		Assert.That(
			new Body(method, 0, new Body(method).AddVariable("str", new Text(method, "Hello"))).
				AddVariable("num", new Number(method, 5)).FindVariableValue("str"),
			Is.EqualTo(new Text(method, "Hello")));

	[Test]
	public void CannotUseVariableFromLowerScope() =>
		Assert.That(() => ParseExpression("if bla is 5", "\tconstant abc = \"abc\"", "log.Write(abc)"),
			Throws.InstanceOf<Body.IdentifierNotFound>().With.Message.StartWith("abc"));

	[Test]
	public void UnknownVariable() =>
		Assert.That(() => ParseExpression("if bla is 5", "\tlog.Write(unknownVariable)"),
			Throws.InstanceOf<Body.IdentifierNotFound>().With.Message.StartWith("unknownVariable"));

	[Test]
	public void CannotAccessAnotherMethodVariable()
	{
		var program = new Type(new Package(nameof(CannotAccessAnotherMethodVariable)),
			new TypeLines(nameof(CannotAccessAnotherMethodVariable),
				// @formatter:off
				"has log",
				"Run",
				"\tconstant number = 5",
				"Add",
				"\tlog.Write(number)")).ParseMembersAndMethods(new MethodExpressionParser());
		// @formatter:on
		Assert.That(
			() => program.Methods[1].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Body.IdentifierNotFound>().With.Message.StartWith("number"));
	}

	[Test]
	public void IfHasDifferentScopeThanMethod() =>
		Assert.That(ParseExpression("if bla is 5", "\tconstant abc = \"abc\"", "\tlog.Write(abc)"),
			Is.EqualTo(new If(GetCondition(), CreateThenBlock())));

	private Expression CreateThenBlock()
	{
		var body = new Body(method);
		var expressions = new Expression[2];
		expressions[0] = new ConstantDeclaration(body, "abc", new Text(method, "abc"));
		var arguments = new Expression[] { new VariableCall("abc", body.FindVariableValue("abc")!) };
		expressions[1] = new MethodCall(member.Type.GetMethod("Write", arguments),
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
				"\tlog.Write(ifText)",
				"else",
				"\tlog.Write(ifText)"),
				// @formatter:on
			Throws.InstanceOf<Body.IdentifierNotFound>().With.Message.StartWith("ifText"));

	[Test]
	public void MissingThenDueToIncorrectChildBodyStart() =>
		Assert.That(() => ParseExpression(
				"if bla is 5",
				"constant abc = \"abc\"",
				"\tlog.Write(abc)"),
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
			"\tlog.Write(abc)") as If;
		var variableCall =
			((ifExpression?.Then as Body)?.Expressions[1] as MethodCall)?.Arguments[0] as VariableCall;
		Assert.That(variableCall?.CurrentValue.ToString(), Is.EqualTo("\"abc\""));
	}

	[Test]
	public void DuplicateVariableNameFound() =>
		Assert.That(() => ParseExpression("if bla is 5", "\tconstant abc = 5", "\tconstant abc = 5"),
			Throws.InstanceOf<Body.ValueIsNotMutableAndCannotBeChanged>().With.Message.StartsWith("abc"));

	[Test]
	public void DuplicateVariableInLowerScopeIsNotAllowed() =>
		Assert.That(
			() => ParseExpression("if bla is 5", "\tconstant outerScope = \"abc\"", "\tif bla is 5.0",
				"\t\tconstant outerScope = 5"),
			Throws.InstanceOf<Body.ValueIsNotMutableAndCannotBeChanged>().With.Message.StartsWith("outerScope"));

	[Test]
	public void ChildBodyReturnsFromThreeTabsToOneDirectly()
	{
		var program = new Type(new Package(nameof(ChildBodyReturnsFromThreeTabsToOneDirectly)),
			new TypeLines(nameof(ChildBodyReturnsFromThreeTabsToOneDirectly),
                // @formatter:off
                "has log",
                "Run",
                "\tconstant number = 5",
                "\tfor Range(1, number)",
                "\t\tif index is number",
                "\t\t\tconstant current = index",
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
}