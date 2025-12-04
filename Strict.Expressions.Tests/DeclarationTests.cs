using Strict.Language.Tests;

namespace Strict.Expressions.Tests;

public class DeclarationTests : TestExpressions
{
	[SetUp]
	public void CreateParserAndPackage() => parser = new MethodExpressionParser();

	private ExpressionParser parser = null!;
	private static readonly Package Package = new(nameof(DeclarationTests));

	[Test]
	public void MissingConstantValue() =>
		Assert.That(() => ParseExpression("constant number"),
			Throws.InstanceOf<Declaration.MissingAssignmentValueExpression>());

	[Test]
	public void ParseNumber()
	{
		var body = (Body)ParseExpression("constant number = 5", "number");
		var assignment = (Declaration)body.Expressions[0];
		Assert.That(assignment, Is.EqualTo(new Declaration(new Body(method), nameof(number), number)));
		Assert.That(assignment.Value.ReturnType, Is.EqualTo(number.ReturnType));
		Assert.That(((Number)assignment.Value).ToString(), Is.EqualTo("5"));
		Assert.That(body.Expressions[1], Is.InstanceOf<VariableCall>());
	}

	[Test]
	public void ParseText()
	{
		var body = (Body)ParseExpression("constant value = \"Hey\"", "value");
		var expression = (Declaration)body.Expressions[0];
		Assert.That(expression.Name, Is.EqualTo("value"));
		Assert.That(expression.Value.ToString(), Is.EqualTo("\"Hey\""));
		Assert.That(body.Expressions[1], Is.InstanceOf<VariableCall>());
	}

	[Test]
	public void AssignmentToString()
	{
		var body = (Body)ParseExpression("constant sum = 5 + 3", "sum");
		var expression = (Declaration)body.Expressions[0];
		Assert.That(expression.Name, Is.EqualTo("sum"));
		Assert.That(expression.Value.ToString(), Is.EqualTo("5 + 3"));
		Assert.That(expression.ToString(), Is.EqualTo("constant sum = 5 + 3"));
	}

	[Test]
	public void AssignmentWithNestedBinary()
	{
		var body = (Body)ParseExpression("constant result = ((5 + 3) * 2 - 5) / 6", "result");
		var expression = (Declaration)body.Expressions[0];
		Assert.That(expression.Name, Is.EqualTo("result"));
		Assert.That(expression.Value, Is.InstanceOf<Binary>());
		var rightExpression = (Number)((Binary)expression.Value).Arguments[0];
		Assert.That(rightExpression.Data, Is.EqualTo(6));
	}

	[Test]
	public void AssignmentWithListAddition()
	{
		var body = (Body)ParseExpression("constant numbers = (1, 2, 3) + 6", "numbers");
		var expression = (Declaration)body.Expressions[0];
		Assert.That(expression.Name, Is.EqualTo("numbers"));
		Assert.That(expression.ReturnType.Name,
			Is.EqualTo(Base.List + "(" + nameof(TestPackage) + "." + Base.Number + ")"));
		Assert.That(expression.Value, Is.InstanceOf<Binary>());
		var leftExpression = ((Binary)expression.Value).Instance!;
		Assert.That(leftExpression.ReturnType.Name,
			Is.EqualTo(Base.List + "(" + nameof(TestPackage) + "." + Base.Number + ")"));
	}

	[Test]
	public void NotAssignment()
	{
		var body = (Body)ParseExpression("constant inverted = not true", "inverted");
		var expression = (Declaration)body.Expressions[0];
		Assert.That(expression.Name, Is.EqualTo("inverted"));
		Assert.That(expression.Value, Is.InstanceOf<Not>());
		Assert.That(expression.Value.ToString(), Is.EqualTo("not true"));
		var rightExpression = (expression.Value as Not)!.Instance as Boolean;
		Assert.That(rightExpression!.Data, Is.EqualTo(true));
	}

	[Test]
	public void InvalidNotAssignment() =>
		Assert.That(() => ParseExpression("constant inverted = not 5"),
			Throws.InnerException.InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>());

	[Test]
	public void OnlyNotIsValidUnaryOperator() =>
		Assert.That(() => ParseExpression("constant inverted = + true"),
			Throws.InstanceOf<InvalidOperatorHere>());

	[Test]
	public void IncompleteAssignment() =>
		Assert.That(() => ParseExpression("constant sum = 5 +"),
			Throws.InstanceOf<InvalidOperatorHere>());

	[Test]
	public void IdentifierMustBeValidWord() =>
		Assert.That(() => ParseExpression("constant number5 = 5"),
			Throws.InnerException.InstanceOf<Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers>());

	[Test]
	public void AssignmentGetHashCode()
	{
		var body = (Body)ParseExpression("constant value = 1", "value");
		var assignment = (Declaration)body.Expressions[0];
		Assert.That(assignment.GetHashCode(),
			Is.EqualTo(assignment.Name.GetHashCode() ^ assignment.Value.GetHashCode()));
	}

	[Test]
	public void LetWithoutVariableNameCannotParse() =>
		Assert.That(() => ParseExpression("constant 5"),
			Throws.InstanceOf<Declaration.MissingAssignmentValueExpression>());

	[Test]
	public void LetWithoutValueCannotParse() =>
		Assert.That(() => ParseExpression("constant value"),
			Throws.InstanceOf<Declaration.MissingAssignmentValueExpression>());

	[Test]
	public void LetWithoutExpressionCannotParse() =>
		Assert.That(() => ParseExpression("constant value = abc"),
			Throws.InstanceOf<Body.IdentifierNotFound>().With.Message.Contain("abc"));

	[Test]
	public void AssignmentWithMethodCall()
	{
		// @formatter:off
		var program = new Type(Package,
			new TypeLines(nameof(AssignmentWithMethodCall),
				"has logger",
				"MethodToCall Text",
				"\t\"Hello World\"",
				"Run",
				"\tconstant result = MethodToCall",
				"\tresult is Text")).ParseMembersAndMethods(parser);
		Assert.That(program.Methods[0].ToString(), Is.EqualTo("MethodToCall Text"));
		var body = (Body)program.Methods[1].GetBodyAndParseIfNeeded();
		Assert.That(body.Expressions[0].ToString(), Is.EqualTo("constant result = MethodToCall"));
	}

	[Test]
	public void LocalMethodCallShouldHaveCorrectReturnType()
	{
		var program = new Type(type.Package,
			new TypeLines(nameof(LocalMethodCallShouldHaveCorrectReturnType),
				"has logger",
				"LocalMethod Text",
				"\t\"Hello World\"",
				"Run",
				"\t\"Random Text\"")).ParseMembersAndMethods(parser);
		Assert.That(program.Methods[0].ReturnType.Name, Is.EqualTo(Base.Text));
	}

	[Test]
	public void LetAssignmentWithConstructorCall() =>
		Assert.That(
			((Declaration)((Body)new Type(Package,
					new TypeLines(nameof(LetAssignmentWithConstructorCall), "has logger",
						"Run",
						"\tconstant file = File(\"test.txt\")",
						"\tfile is File")).ParseMembersAndMethods(parser).Methods[0].
				GetBodyAndParseIfNeeded()).Expressions[0]).Value.ToString(), Is.EqualTo("File(\"test.txt\")"));
}