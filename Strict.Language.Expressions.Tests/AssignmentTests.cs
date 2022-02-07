using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public class AssignmentTests : TestExpressions
{
	[Test]
	public void MultipleStatementsAreNotAllowedHere() =>
		Assert.That(() => ParseExpression(method, "let number = 5\nlet other = 3"),
			Throws.Exception.InstanceOf<MultipleExpressionsGiven>());

	[Test]
	public void ParseNumber()
	{
		var assignment = (Assignment)ParseExpression(method, "let number = 5");
		Assert.That(assignment,
			Is.EqualTo(new Assignment(new Identifier(nameof(number), number.ReturnType), number)));
		Assert.That(assignment.Value.ReturnType, Is.EqualTo(number.ReturnType));
		Assert.That(((Number)assignment.Value).ToString(), Is.EqualTo("5"));
	}

	[Test]
	public void ParseText()
	{
		const string Input = "let value = \"Hey\"";
		var expression = (Assignment)ParseExpression(method, Input);
		Assert.That(expression.Name.ToString(), Is.EqualTo("value"));
		Assert.That(expression.Value.ToString(), Is.EqualTo("\"Hey\""));
		Assert.That(expression.ToString(), Is.EqualTo(Input));
	}

	[Test]
	public void AssignmentToString()
	{
		const string Input = "let sum = 5 + 3";
		var expression = (Assignment)ParseExpression(method, Input);
		Assert.That(expression.Name.ToString(), Is.EqualTo("sum"));
		Assert.That(expression.Value.ToString(), Is.EqualTo("5 + 3"));
		Assert.That(expression.ToString(), Is.EqualTo(Input));
	}

	[Test]
	public void IncompleteAssignment()
	{
		const string Input = "let sum = 5 + ";
		Assert.That(() => ParseExpression(method, Input),
			Throws.Exception.InstanceOf<EmptyExpression>());
	}

	[Test]
	public void IdentifierMustBeValidWord() =>
		Assert.That(() => ParseExpression(method, "let number5 = 5"),
			Throws.Exception.InstanceOf<Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers>());

	[Test]
	public void AssignmentGetHashCode()
	{
		var assignment = (Assignment)ParseExpression(method, "let value = 1");
		Assert.That(assignment.GetHashCode(),
			Is.EqualTo(assignment.Name.GetHashCode() ^ assignment.Value.GetHashCode()));
	}

	[Test]
	public void LetWithoutVariableNameCannotParse() =>
		Assert.That(() => ParseExpression(method, "let 5"),
			Throws.Exception.InstanceOf<Assignment.IncompleteLet>());

	[Test]
	public void LetWithoutExpressionCannotParse() =>
		Assert.That(() => ParseExpression(method, "let value = abc"),
			Throws.Exception.InstanceOf<MemberCall.MemberNotFound>().With.Message.Contain("abc"));
}