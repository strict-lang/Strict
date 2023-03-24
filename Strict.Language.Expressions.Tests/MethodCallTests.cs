using NUnit.Framework;
using static Strict.Language.Type;

namespace Strict.Language.Expressions.Tests;

public sealed class MethodCallTests : TestExpressions
{
	[SetUp]
	public void AddComplexMethods()
	{
		((List<Method>)type.Methods).Add(new Method(type, 0, this,
			new[] { "ComplexMethod(numbers, add Number) Number", "\t1" }));
		((List<Method>)type.Methods).Add(new Method(type, 0, this,
			new[] { "ComplexMethod(texts) Texts", "\t1" }));
		((List<Method>)type.Methods).Add(new Method(type, 0, this,
			new[] { "ComplexMethod(numbers) Texts", "\t1" }));
	}

	[Test]
	public void ParseLocalMethodCall() =>
		ParseAndCheckOutputMatchesInput("Run", new MethodCall(type.Methods[0]));

	[Test]
	public void ParseCallWithArgument() =>
		ParseAndCheckOutputMatchesInput("log.Write(bla)",
			new MethodCall(member.Type.Methods[0], new MemberCall(null, member),
				new[] { new MemberCall(null, bla) }));

	[Test]
	public void ParseCallWithTextArgument() =>
		ParseAndCheckOutputMatchesInput("log.Write(\"Hi\")",
			new MethodCall(member.Type.Methods[0], new MemberCall(null, member),
				new[] { new Text(type, "Hi") }));

	[Test]
	public void ParseWithMissingArgument() =>
		Assert.That(() => ParseExpression("log.Write"),
			Throws.InstanceOf<ArgumentsDoNotMatchMethodParameters>().With.Message.StartsWith(
				"No arguments does not match these method(s):\nWrite(generic TestPackage.Generic)"));

	[Test]
	public void ParseWithTooManyArguments() =>
		Assert.That(() => ParseExpression("log.Write(1, 2)"),
			Throws.InstanceOf<ArgumentsDoNotMatchMethodParameters>().With.Message.
				StartsWith("Arguments: 1 TestPackage.Number, 2 TestPackage.Number do not match"));

	[Test]
	public void ParseWithInvalidExpressionArguments() =>
		Assert.That(() => ParseExpression("log.Write(g9y53)"),
			Throws.InstanceOf<UnknownExpressionForArgument>().With.Message.
				StartsWith("g9y53 (argument 0)"));

	[Test]
	public void EmptyBracketsAreNotAllowed() =>
		Assert.That(() => ParseExpression("log.NotExisting()"),
			Throws.InstanceOf<List.EmptyListNotAllowed>());

	[Test]
	public void MethodNotFound() =>
		Assert.That(() => ParseExpression("log.NotExisting"),
			Throws.InstanceOf<MemberOrMethodNotFound>());

	[Test]
	public void ArgumentsDoNotMatchMethodParameters() =>
		Assert.That(() => ParseExpression("Character(\"Hi\")"),
			Throws.InstanceOf<ArgumentsDoNotMatchMethodParameters>());

	[Test]
	public void ParseCallWithUnknownMemberCallArgument() =>
		Assert.That(() => ParseExpression("log.Write(log.unknown)"),
			Throws.InstanceOf<MemberOrMethodNotFound>().With.Message.
				StartsWith("unknown in TestPackage.Log"));

	[Test]
	public void MethodCallMembersMustBeWords() =>
		Assert.That(() => ParseExpression("g9y53.Write"), Throws.InstanceOf<MemberOrMethodNotFound>());

	[Test]
	public void UnknownExpressionForArgumentException() =>
		Assert.That(() => ParseExpression("ComplexMethod(true)"),
			Throws.InstanceOf<ArgumentsDoNotMatchMethodParameters>().With.Message.
				StartsWith("Argument: true "));

	[Test]
	public void ListTokensAreNotSeparatedByCommaException() =>
		Assert.That(() => ParseExpression("ComplexMethod((\"1 + 5\" 5, \"5 + 5\"))"),
			Throws.InstanceOf<ListTokensAreNotSeparatedByComma>());

	[Test]
	public void SimpleFromMethodCall() =>
		Assert.That(ParseExpression("Character(7)"),
			Is.EqualTo(CreateFromMethodCall(type.GetType(Base.Character), new Number(type, 7))));

	[TestCase("Character(5)")]
	[TestCase("Range(0, 10)")]
	[TestCase("Range(0, 10).Length")]
	public void FromExample(string fromMethodCall) =>
		Assert.That(ParseExpression(fromMethodCall).ToString(), Is.EqualTo(fromMethodCall));

	[Test]
	public void MakeSureMutableTypeMethodsAreNotModified()
	{
		var expression = ParseExpression("mutable variable = 7");
		Assert.That(type.GetType(Base.Mutable).Methods.Count, Is.EqualTo(0));
		Assert.That(expression is MutableDeclaration, Is.True);
		Assert.That(((MutableDeclaration)expression).Value.IsMutable, Is.True);
	}

	[Test]
	public void FromExampleFailsOnImproperParameters() =>
		Assert.That(() => ParseExpression("Range(1, 2, 3, 4)"),
			Throws.InstanceOf<NoMatchingMethodFound>());

	[TestCase("ComplexMethod((1), 2)")]
	[TestCase("ComplexMethod((1, 2, 3))")]
	[TestCase("ComplexMethod((1, 2, 3) + (4, 5), 7)")]
	[TestCase("ComplexMethod((1, 2, 3) + (4, 5), ComplexMethod((1, 2, 3), 4))")]
	[TestCase("ComplexMethod((\"1 + 5\", \"5 + 5\"))")]
	public void FindRightMethodCall(string methodCall) =>
		Assert.That(ParseExpression(methodCall).ToString(), Is.EqualTo(methodCall));

	[Test]
	public void IsMethodPublic() =>
		Assert.That((ParseExpression("Run") as MethodCall)?.Method.IsPublic, Is.True);

	[Test]
	public void ValueMustHaveCorrectType()
	{
		var program = new Type(type.Package, new TypeLines(
				nameof(ValueMustHaveCorrectType),
				"has log",
				"has Number",
				$"Dummy(dummy Number) {nameof(ValueMustHaveCorrectType)}",
				"\tconstant result = value",
				"\tresult")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(
			((Body)program.Methods[0].GetBodyAndParseIfNeeded()).FindVariableValue("value")?.ReturnType,
			Is.EqualTo(program));
	}

	[Test]
	public void CanAccessThePropertiesOfValue()
	{
		var program = new Type(type.Package, new TypeLines(
				nameof(CanAccessThePropertiesOfValue),
				"has log",
				"has Number",
				"has myMember Text",
				"Dummy(dummy Number) Text",
				"\tconstant result = value.myMember",
				"\tresult")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.FindVariableValue("value")?.ReturnType, Is.EqualTo(program));
		Assert.That(body.FindVariableValue("result")?.ReturnType.Name, Is.EqualTo("Text"));
	}

	[TestCase("ProgramWithHas", "numbers",
		"has numbers",
		"Dummy",
		"\tconstant instanceWithNumbers = ProgramWithHas((1, 2, 3))")]
	[TestCase("ProgramWithPublicMember",
		"texts", "has Texts",
		"Dummy",
		"\tconstant instanceWithTexts = ProgramWithPublicMember((\"1\", \"2\", \"3\"))")]
	public void ParseConstructorCallWithList(string programName, string expected, params string[] code)
	{
		var program = new Type(type.Package, new TypeLines(
				programName,
				code)).
			ParseMembersAndMethods(new MethodExpressionParser());
		var assignment = (ConstantDeclaration)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(((MethodCall)assignment.Value).Method.Parameters[0].Name, Is.EqualTo(expected));
	}

	[Test]
	public void TypeImplementsGenericTypeWithLength()
	{
		new Type(type.Package,
			new TypeLines("HasLengthImplementation",
				"has HasLength",
				"has boolean",
				"Length Number",
				"\tvalue")).ParseMembersAndMethods(new MethodExpressionParser());
		var program = new Type(type.Package,
			new TypeLines(nameof(TypeImplementsGenericTypeWithLength),
				"has log", //unused member should be removed later when we allow class without members
				"GetLengthSquare(type HasLength) Number",
				"\ttype.Length * type.Length",
				"Dummy",
				"\tconstant countOfFive = HasLengthImplementation(true)",
				"\tconstant lengthSquare = GetLengthSquare(countOfFive)")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(program.Methods[1].GetBodyAndParseIfNeeded().ToString(),
			Is.EqualTo(
				"constant countOfFive = HasLengthImplementation(true)\r\nconstant lengthSquare = GetLengthSquare(countOfFive)"));
	}

	[Test]
	public void MutableCanUseChildMethods()
	{
		var program = new Type(type.Package,
			new TypeLines(nameof(MutableCanUseChildMethods),
				"has log",
				"Dummy Number",
				"\tconstant mutableNumber = 5",
				"\tmutableNumber + 10")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(program.Methods[0].GetBodyAndParseIfNeeded().ToString(),
			Is.EqualTo("constant mutableNumber = 5\r\nmutableNumber + 10"));
	}

	[Test]
	public void ConstructorCallWithMethodCall()
	{
		var program = new Type(type.Package,
			new TypeLines("ArithmeticFunction",
				"has numbers",
				"from(first Number, second Number)",
				"\tnumbers = (first, second)",
				"Calculate(text) Number",
				"\tArithmeticFunction(10, 5).Calculate(\"add\") is 15",
				"\t1")).ParseMembersAndMethods(new MethodExpressionParser());
		program.Methods[1].GetBodyAndParseIfNeeded();
	}

	[Test]
	public void RecursiveStackOverflow()
	{
		var program = new Type(type.Package,
			new TypeLines("RecursiveStackOverflow",
				"has number",
				"AddFiveWithInput Number",
				"\tRecursiveStackOverflow(10).AddFiveWithInput is 15",
				"\tRecursiveStackOverflow(10).AddFiveWithInput",
				"\tnumber + 5")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<RecursiveCallCausesStackOverflow>());
	}

	[Test]
	public void NestedMethodCall()
	{
		var program = new Type(type.Package,
				new TypeLines(nameof(NestedMethodCall), "has log", "Run",
					"\tFile(\"fileName\").Write(\"someText\")", "\ttrue")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.Expressions[0], Is.InstanceOf<MethodCall>());
		Assert.That(((MethodCall)body.Expressions[0]).Method.Name, Is.EqualTo("Write"));
		Assert.That(((MethodCall)body.Expressions[0]).ToString(), Is.EqualTo("File(\"fileName\").Write(\"someText\")"));
		Assert.That(((MethodCall)body.Expressions[0]).Instance?.ToString(),
			Is.EqualTo("File(\"fileName\")"));
	}

	[Test]
	public void MethodCallAsMethodParameter()
	{
		var program = new Type(type.Package,
				new TypeLines(nameof(MethodCallAsMethodParameter),
					"has log",
					"AppendFiveWithInput(number) Number",
					"\tAppendFiveWithInput(AppendFiveWithInput(5)) is 15",
					"\tnumber + 5")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.Expressions[0].ToString(),
			Is.EqualTo("AppendFiveWithInput(AppendFiveWithInput(5)) is 15"));
		Assert.That(((MethodCall)((MethodCall)body.Expressions[0]).Instance!).Arguments[0].ToString(),
			Is.EqualTo("AppendFiveWithInput(5)"));
	}

	[Test]
	public void TypeCanBeAutoInitialized()
	{
		new Type(type.Package,
				new TypeLines(nameof(TypeCanBeAutoInitialized),
					"has log",
					"AddFiveWithInput(number) Number",
					"\tAddFiveWithInput(AddFiveWithInput(5)) is 15",
					"\tnumber + 5")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var consumingType = new Type(type.Package,
				new TypeLines("AutoInitializedTypeConsumer",
					"has typeCanBeAutoInitialized",
					"GetResult(number) Number",
					"\tGetResult(10) is 15",
					"\ttypeCanBeAutoInitialized.AddFiveWithInput(number)")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var body = (Body)consumingType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(consumingType.Members[0].Type.Name, Is.EqualTo("TypeCanBeAutoInitialized"));
		Assert.That(((MethodCall)body.Expressions[1]).Instance?.ReturnType.Name, Is.EqualTo("TypeCanBeAutoInitialized"));
	}

	[Test]
	public void TypeCannotBeAutoInitialized()
	{
		new Type(type.Package,
				new TypeLines(nameof(TypeCannotBeAutoInitialized),
					"has number",
					"AddFiveWithInput Number",
					"\tTypeCannotBeAutoInitialized(10).AddFiveWithInput is 15",
					"\tnumber + 5")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var consumer = new Type(type.Package,
				new TypeLines("ConsumingType",
					"has log",
					"GetResult(number) Number",
					"\tGetResult(10) is 15",
					"\tconstant instance = TypeCannotBeAutoInitialized(number)",
					"\tinstance.AddFiveWithInput")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var body = (Body)consumer.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(((ConstantDeclaration)body.Expressions[1]).Value.ReturnType.Name, Is.EqualTo("TypeCannotBeAutoInitialized"));
	}
}