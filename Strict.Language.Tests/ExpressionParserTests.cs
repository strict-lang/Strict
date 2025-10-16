namespace Strict.Language.Tests;

public class ExpressionParserTests : ExpressionParser
{
	[SetUp]
	public void CreateType() =>
		type = new Type(TestPackage.Instance, new MockRunTypeLines()).ParseMembersAndMethods(this);

	private Type type = null!;

	[TearDown]
	public void TearDown() => type.Dispose();

	[Test]
	public void ParsingHappensAfterCallingGetBodyAndParseIfNeeded()
	{
		Assert.That(parseWasCalled, Is.False);
		Assert.That(type.Methods[0].GetBodyAndParseIfNeeded(), Is.InstanceOf<Expression>());
		Assert.That(parseWasCalled, Is.True);
		Assert.That(type.Methods[0].GetBodyAndParseIfNeeded().ReturnType,
			Is.EqualTo(type.Methods[0].ReturnType));
	}

	private bool parseWasCalled;

	public class TestExpression(Type returnType) : Expression(returnType)
	{
		public override bool IsConstant => false;
	}

	public override Expression ParseLineExpression(Body body, ReadOnlySpan<char> line)
	{
		parseWasCalled = true;
		return new MethodCall(body.Method);
	}

	//ncrunch: no coverage start, not the focus here
	public override Expression ParseExpression(Body body, ReadOnlySpan<char> text,
		bool isMutable = false) =>
		new Value(type.GetType(Base.Boolean), int.TryParse(text, out _), isMutable);

	public override List<Expression> ParseListArguments(Body body, ReadOnlySpan<char> text) => null!;

	public override bool IsVariableMutated(Body body, string variableName) => false;
	//ncrunch: no coverage end

	[Test]
	public void CompareExpressions()
	{
		var expression = new TestExpression(type);
		Assert.That(expression, Is.EqualTo(new TestExpression(type)));
		Assert.That(expression.GetHashCode(), Is.EqualTo(new TestExpression(type).GetHashCode()));
		Assert.That(new TestExpression(type.Methods[0].ReturnType),
			Is.Not.EqualTo(new TestExpression(type)));
		Assert.That(expression.Equals((object)new TestExpression(type)), Is.True);
	}

	[Test]
	public void EmptyLineIsNotValidInMethods() =>
		Assert.That(() => new Method(type, 0, this, ["Run", ""]),
			Throws.InstanceOf<TypeParser.EmptyLineIsNotAllowed>());

	[Test]
	public void NoIndentationIsNotValidInMethods() =>
		Assert.That(() => new Method(type, 0, this, ["Run", "abc"]),
			Throws.InstanceOf<Method.InvalidIndentation>());

	[Test]
	public void TooMuchIndentationIsNotValidInMethods() =>
		Assert.That(() => new Method(type, 0, this, ["Run", new string('\t', 4)]),
			Throws.InstanceOf<Method.InvalidIndentation>());

	[Test]
	public void ExtraWhitespacesAtBeginningOfLineAreNotAllowed() =>
		Assert.That(() => new Method(type, 0, this, ["Run", "\t constant abc = 3"]),
			Throws.InstanceOf<TypeParser.ExtraWhitespacesFoundAtBeginningOfLine>());

	[Test]
	public void ExtraWhitespacesAtEndOfLineAreNotAllowed() =>
		Assert.That(() => new Method(type, 0, this, ["Run", "\tconstant abc = 3 "]),
			Throws.InstanceOf<TypeParser.ExtraWhitespacesFoundAtEndOfLine>());

	[Test]
	public void GetSingleLine()
	{
		var method = new Method(type, 0, this, ["Run", MethodTests.ConstantNumber]);
		Assert.That(method.lines, Has.Count.EqualTo(2));
		Assert.That(method.lines[0], Is.EqualTo("Run"));
		Assert.That(method.lines[1], Is.EqualTo(MethodTests.ConstantNumber));
	}

	[Test]
	public void GetMultipleLines()
	{
		var method = new Method(type, 0, this, ["Run", MethodTests.ConstantNumber, MethodTests.ConstantOther]);
		Assert.That(method.lines, Has.Count.EqualTo(3));
		Assert.That(method.lines[1], Is.EqualTo(MethodTests.ConstantNumber));
		Assert.That(method.lines[2], Is.EqualTo(MethodTests.ConstantOther));
	}

	[Test]
	public void GetNestedLines()
	{
		var method = new Method(type, 0, this, MethodTests.NestedMethodLines);
		Assert.That(method.lines, Has.Length.EqualTo(5));
		Assert.That(method.lines[1], Is.EqualTo(MethodTests.ConstantNumber));
		Assert.That(method.lines[2], Is.EqualTo("	if bla is 5"));
		Assert.That(method.lines[3], Is.EqualTo("		return true"));
		Assert.That(method.lines[4], Is.EqualTo("	false"));
	}
}