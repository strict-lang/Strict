using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public class AssignmentTests : TestExpressions
{
	[Test]
	public void MultipleStatementsAreNotAllowedHere() =>
		Assert.That(() => ParseExpression("let number = 5", "let other = 3"),
			Throws.Exception.InstanceOf<MultipleExpressionsGiven>());

	[Test]
	public void ParseNumber()
	{
		var assignment = (Assignment)ParseExpression("let number = 5");
		Assert.That(assignment, Is.EqualTo(new Assignment(method, nameof(number), number)));
		Assert.That(assignment.Value.ReturnType, Is.EqualTo(number.ReturnType));
		Assert.That(((Number)assignment.Value).ToString(), Is.EqualTo("5"));
	}

	[Test]
	public void ParseText()
	{
		const string Input = "let value = \"Hey\"";
		var expression = (Assignment)ParseExpression(Input);
		Assert.That(expression.Name, Is.EqualTo("value"));
		Assert.That(expression.Value.ToString(), Is.EqualTo("\"Hey\""));
		Assert.That(expression.ToString(), Is.EqualTo(Input));
	}

	[Test]
	public void AssignmentToString()
	{
		const string Input = "let sum = 5 + 3";
		var expression = (Assignment)ParseExpression(Input);
		Assert.That(expression.Name, Is.EqualTo("sum"));
		Assert.That(expression.Value.ToString(), Is.EqualTo("5 + 3"));
		Assert.That(expression.ToString(), Is.EqualTo(Input));
	}

	[Test]
	public void AssignmentWithNestedBinary()
	{
		const string Input = "let result = ((5 + 3) * 2 - 5) / 6";
		var expression = (Assignment)ParseExpression(Input);
		Assert.That(expression.Name, Is.EqualTo("result"));
		Assert.That(expression.Value, Is.InstanceOf<Binary>());
		var rightExpression = (Number)((Binary)expression.Value).Arguments[0];
		Assert.That(rightExpression.Data, Is.EqualTo(6));
	}

	[Test]
	public void NotAssignment()
	{
		const string Input = "let inverted = not true";
		var expression = (Assignment)ParseExpression(Input);
		Assert.That(expression.Name, Is.EqualTo("inverted"));
		Assert.That(expression.Value, Is.InstanceOf<Not>());
		Assert.That(expression.Value.ToString(), Is.EqualTo("not true"));
		var rightExpression = (expression.Value as Not)!.Instance as Boolean;
		Assert.That(rightExpression!.Data, Is.EqualTo(true));
	}

	[Test]
	public void OnlyNotIsValidUnaryOperator() =>
		Assert.That(() => ParseExpression("let inverted = + true"),
			Throws.Exception.InstanceOf<InvalidOperatorHere>());

	[Test]
	public void IncompleteAssignment() =>
		Assert.That(() => ParseExpression("let sum = 5 +"),
			Throws.Exception.InstanceOf<InvalidOperatorHere>());

	[Test]
	public void IdentifierMustBeValidWord() =>
		Assert.That(() => ParseExpression("let number5 = 5"),
			Throws.Exception.InstanceOf<Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers>());

	[Test]
	public void AssignmentGetHashCode()
	{
		var assignment = (Assignment)ParseExpression("let value = 1");
		Assert.That(assignment.GetHashCode(),
			Is.EqualTo(assignment.Name.GetHashCode() ^ assignment.Value.GetHashCode()));
	}

	[Test]
	public void LetWithoutVariableNameCannotParse() =>
		Assert.That(() => ParseExpression("let 5"),
			Throws.Exception.InstanceOf<Assignment.IncompleteLet>());

	[Test]
	public void LetWithoutValueCannotParse() =>
		Assert.That(() => ParseExpression("let value"),
			Throws.Exception.InstanceOf<Assignment.IncompleteLet>());

	[Test]
	public void LetWithoutExpressionCannotParse() =>
		Assert.That(() => ParseExpression("let value = abc"),
			Throws.Exception.InstanceOf<UnknownExpression>().With.Message.Contain("abc"));

	[Test]
	[Ignore("TODO: Not yet done")]
	public void AssignmentWithArguments()
	{
		const string Code = "has input = Text(5)";
		// this is a bit strange, why would we need assigmentType here, it is not used by Code yet?
		//var assignmentType =
		new Type(type.Package,
			new TypeLines("Assignment", "has number", "has file = \"test.txt\"", "Run", "\tfile.Write(number)")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(((Assignment)ParseExpression(Code)).ToString(), Is.EqualTo(Code));
	}
}