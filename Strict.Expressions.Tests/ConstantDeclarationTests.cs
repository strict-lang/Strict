using Strict.Language.Tests;

namespace Strict.Expressions.Tests;

public class ConstantDeclarationTests : TestExpressions
{
	[SetUp]
	public void CreateParserAndPackage() => parser = new MethodExpressionParser();

	private ExpressionParser parser = null!;
	private static readonly Package Package = new(nameof(ConstantDeclarationTests));

	[Test]
	public void MissingConstantValue() =>
		Assert.That(() => ParseExpression("constant number"),
			Throws.InstanceOf<ConstantDeclaration.MissingAssignmentValueExpression>());

	[Test]
	public void ParseNumber()
	{
		var assignment = (ConstantDeclaration)ParseExpression("constant number = 5");
		Assert.That(assignment, Is.EqualTo(new ConstantDeclaration(new Body(method), nameof(number), number)));
		Assert.That(assignment.Value.ReturnType, Is.EqualTo(number.ReturnType));
		Assert.That(((Number)assignment.Value).ToString(), Is.EqualTo("5"));
	}

	[Test]
	public void ParseText()
	{
		const string Input = "constant value = \"Hey\"";
		var expression = (ConstantDeclaration)ParseExpression(Input);
		Assert.That(expression.Name, Is.EqualTo("value"));
		Assert.That(expression.Value.ToString(), Is.EqualTo("\"Hey\""));
		Assert.That(expression.ToString(), Is.EqualTo(Input));
	}

	[Test]
	public void AssignmentToString()
	{
		const string Input = "constant sum = 5 + 3";
		var expression = (ConstantDeclaration)ParseExpression(Input);
		Assert.That(expression.Name, Is.EqualTo("sum"));
		Assert.That(expression.Value.ToString(), Is.EqualTo("5 + 3"));
		Assert.That(expression.ToString(), Is.EqualTo(Input));
	}

	[Test]
	public void AssignmentWithNestedBinary()
	{
		const string Input = "constant result = ((5 + 3) * 2 - 5) / 6";
		var expression = (ConstantDeclaration)ParseExpression(Input);
		Assert.That(expression.Name, Is.EqualTo("result"));
		Assert.That(expression.Value, Is.InstanceOf<Binary>());
		var rightExpression = (Number)((Binary)expression.Value).Arguments[0];
		Assert.That(rightExpression.Data, Is.EqualTo(6));
	}

	[Test]
	public void AssignmentWithListAddition()
	{
		const string Input = "constant numbers = (1, 2, 3) + 6";
		var expression = (ConstantDeclaration)ParseExpression(Input);
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
		const string Input = "constant inverted = not true";
		var expression = (ConstantDeclaration)ParseExpression(Input);
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
		var assignment = (ConstantDeclaration)ParseExpression("constant value = 1");
		Assert.That(assignment.GetHashCode(),
			Is.EqualTo(assignment.Name.GetHashCode() ^ assignment.Value.GetHashCode()));
	}

	[Test]
	public void LetWithoutVariableNameCannotParse() =>
		Assert.That(() => ParseExpression("constant 5"),
			Throws.InstanceOf<ConstantDeclaration.MissingAssignmentValueExpression>());

	[Test]
	public void LetWithoutValueCannotParse() =>
		Assert.That(() => ParseExpression("constant value"),
			Throws.InstanceOf<ConstantDeclaration.MissingAssignmentValueExpression>());

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
				"\tconstant result = MethodToCall")).ParseMembersAndMethods(parser);
		Assert.That(program.Methods[0].ToString(), Is.EqualTo("MethodToCall Text"));
		Assert.That(program.Methods[1].GetBodyAndParseIfNeeded().ToString(),
			Is.EqualTo("constant result = MethodToCall"));
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
			((ConstantDeclaration)new Type(Package,
					new TypeLines(nameof(LetAssignmentWithConstructorCall), "has logger",
						"Run",
						"\tconstant file = File(\"test.txt\")")).ParseMembersAndMethods(parser).Methods[0].
				GetBodyAndParseIfNeeded()).Value.ToString(), Is.EqualTo("File(\"test.txt\")"));
}