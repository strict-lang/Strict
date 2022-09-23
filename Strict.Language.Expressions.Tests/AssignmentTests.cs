using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

public class AssignmentTests : TestExpressions
{
	[SetUp]
	public void CreateParserAndPackage()
	{
		parser = new MethodExpressionParser();
		package = new Package(nameof(AssignmentTests));
	}

	private ExpressionParser parser = null!;
	private Package package = null!;

	[Test]
	public void ParseNumber()
	{
		var assignment = (Assignment)ParseExpression("let number = 5");
		Assert.That(assignment, Is.EqualTo(new Assignment(new Body(method), nameof(number), number)));
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
	public void AssignmentWithListAddition()
	{
		const string Input = "let numbers = (1, 2, 3) + 6";
		var expression = (Assignment)ParseExpression(Input);
		Assert.That(expression.Name, Is.EqualTo("numbers"));
		const string ListNumber = Base.List + Base.Number;
		Assert.That(expression.ReturnType.Name, Is.EqualTo(ListNumber));
		Assert.That(expression.Value, Is.InstanceOf<Binary>());
		var leftExpression = ((Binary)expression.Value).Instance!;
		Assert.That(leftExpression.ReturnType.Name, Is.EqualTo(ListNumber));
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
	public void InvalidNotAssignment() =>
		Assert.That(() => ParseExpression("let inverted = not 5"),
			Throws.InstanceOf<Type.NoMatchingMethodFound>());

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
			Throws.Exception.InstanceOf<IdentifierNotFound>().With.Message.Contain("abc"));

	[Test]
	public void AssignmentWithMethodCall()
	{
		// @formatter:off
		var program = new Type(package,
			new TypeLines(nameof(AssignmentWithMethodCall),
				"has log",
				"MethodToCall Text",
				"\t\"Hello World\"",
		"Run",
				"\tlet result = MethodToCall")).ParseMembersAndMethods(parser);
		Assert.That(program.Methods[0].ToString(), Is.EqualTo("MethodToCall Text"));
		Assert.That(program.Methods[1].GetBodyAndParseIfNeeded().ToString(),
			Is.EqualTo("let result = MethodToCall"));
	}

	[Test]
	public void LocalMethodCallShouldHaveCorrectReturnType()
	{
		var program = new Type(type.Package,
			new TypeLines(nameof(LocalMethodCallShouldHaveCorrectReturnType),
				"has log",
				"LocalMethod Text",
				"\t\"Hello World\"",
		"Run",
				"\t\"Random Text\"")).ParseMembersAndMethods(parser);
		Assert.That(program.Methods[0].ReturnType.Name, Is.EqualTo(Base.Text));
	}

	[Test]
	public void LetAssignmentWithConstructorCall() =>
		Assert.That(
			((Assignment)new Type(package,
					new TypeLines(nameof(LetAssignmentWithConstructorCall), "has log",
						"Run",
						"\tlet file = File(\"test.txt\")")).ParseMembersAndMethods(parser).Methods[0].
				GetBodyAndParseIfNeeded()).Value.ToString(), Is.EqualTo("File(\"test.txt\")"));
}