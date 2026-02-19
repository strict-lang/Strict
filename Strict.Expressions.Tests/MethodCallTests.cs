namespace Strict.Expressions.Tests;

public sealed class MethodCallTests : TestExpressions
{
	[SetUp]
	public void AddComplexMethods()
	{
		type.Methods.Add(new Method(type, 0, this, [
			"ComplexMethod(numbers, add Number) Number", "\t1"
		]));
		type.Methods.Add(new Method(type, 0, this, ["ComplexMethod(texts) Texts", "\t1"]));
		type.Methods.Add(new Method(type, 0, this, ["ComplexMethod(numbers) Texts", "\t1"]));
	}

	[Test]
	public void ParseLocalMethodCall() =>
		ParseAndCheckOutputMatchesInput("Run", new MethodCall(type.Methods[0]));

	[Test]
	public void ParseCallWithArgument() =>
		ParseAndCheckOutputMatchesInput("logger.Log(five)", new MethodCall(member.Type.Methods[0],
			new MemberCall(null, member), [
				new MemberCall(null, five)
			]));

	[Test]
	public void ParseCallWithTextArgument() =>
		ParseAndCheckOutputMatchesInput("logger.Log(\"Hi\")",
			new MethodCall(member.Type.Methods[0], new MemberCall(null, member),
				[new Text(type, "Hi")]));

	[Test]
	public void ParseCallWithBinaryArgument() =>
		Assert.That(ParseExpression("(10 / 2).Floor").ToString(), Is.EqualTo("(10 / 2).Floor"));

	[Test]
	public void PrivateMethodWithTooManyArgumentsIsNotMatched()
	{
		type.Methods.Add(new Method(type, 0, this, ["hidden(number) Number", "\t1"]));
		Assert.That(() => ParseExpression("hidden(1, 2)"),
     Throws.InstanceOf<ParsingFailed>());
	}

	[Test]
	public void PrivateMethodWithTooFewArgumentsIsNotMatched()
	{
		type.Methods.Add(new Method(type, 0, this,
			["hiddenWithTwo(number, other Number) Number", "\t1"]));
		Assert.That(() => ParseExpression("hiddenWithTwo(1)"),
     Throws.InstanceOf<ParsingFailed>());
	}

	[Test]
	public void PrivateMethodWithWrongArgumentTypeIsNotMatched()
	{
		type.Methods.Add(new Method(type, 0, this, ["hiddenText(number) Number", "\t1"]));
		Assert.That(() => ParseExpression("hiddenText(\"text\")"),
     Throws.InstanceOf<ParsingFailed>());
	}

	[Test]
	public void PrivateMethodWithMutableParameterRejectsNonListArgument()
	{
		type.Methods.Add(new Method(type, 0, this,
			["hiddenMutable(mutableList Mutable(List)) Number", "\t1"]));
		Assert.That(() => ParseExpression("hiddenMutable(1)"),
			Throws.InstanceOf<ParsingFailed>());
	}

	[Test]
	public void ParseCallWithMemberAccessInNestedArgument()
	{
		var digitsMethod = new Method(type, 0, this, [
			"digits(number) Number",
			"\tdigits((number / 10).Floor) + number % 10"
		]);
		type.Methods.Add(digitsMethod);
    var binary = (Binary)digitsMethod.GetBodyAndParseIfNeeded();
		var methodCall = (MethodCall)binary.Instance!;
		Assert.That(methodCall.Method.Name, Is.EqualTo("digits"));
		Assert.That(methodCall.Arguments[0].ToString(), Is.EqualTo("(number / 10).Floor"));
	}

	[Test]
	public void ParseWithMissingArgument() =>
		Assert.That(() => ParseExpression("logger.Write"),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>().With.Message.Contains("logger.Write"));

	[Test]
	public void ParseWithTooManyArguments() =>
		Assert.That(() => ParseExpression("logger.Log(1, 2)"),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>().With.Message.
				Contains("logger.Log(1, 2)"));

	[Test]
	public void ParseWithInvalidExpressionArguments() =>
		Assert.That(() => ParseExpression("logger.Log(g9y53)"),
			Throws.InstanceOf<UnknownExpressionForArgument>().With.Message.
				StartsWith("g9y53 (argument 0)"));

	[Test]
	public void EmptyBracketsAreNotAllowed() =>
		Assert.That(() => ParseExpression("logger.NotExisting()"),
			Throws.InstanceOf<List.EmptyListNotAllowed>());

	[Test]
	public void MethodNotFound() =>
		Assert.That(() => ParseExpression("logger.NotExisting"),
			Throws.InstanceOf<MemberOrMethodNotFound>());

	[Test]
	public void ArgumentsDoNotMatchMethodParameters() =>
		Assert.That(() => ParseExpression("Character(\"Hi\")"),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>());

	[Test]
	public void ParseCallWithUnknownMemberCallArgument() =>
		Assert.That(() => ParseExpression("logger.Log(logger.unknown)"),
			Throws.InstanceOf<MemberOrMethodNotFound>().With.Message.
				StartsWith("unknown in TestPackage.Logger"));

	[Test]
	public void MethodCallMembersMustBeWords() =>
		Assert.That(() => ParseExpression("g9y53.Write"), Throws.InstanceOf<MemberOrMethodNotFound>());

	[Test]
	public void UnknownExpressionForArgumentException() =>
		Assert.That(() => ParseExpression("ComplexMethod(true)"),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>().With.Message.
				Contains("ComplexMethod(true)"));

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
		var body = (Body)ParseExpression("mutable variable = 7", "variable = variable + 1");
		var expression = body.Expressions[0];
		Assert.That(type.GetType(Base.Mutable).Methods.Count, Is.EqualTo(0));
		Assert.That(expression is Declaration, Is.True);
		Assert.That(((Declaration)expression).IsMutable, Is.True);
	}

	[Test]
	public void FromExampleFailsOnImproperParameters() =>
		Assert.That(() => ParseExpression("Range(1, 2, 3, 4)"),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>());

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
				"has logger",
				"has Number",
				$"Dummy(dummy Number) {nameof(ValueMustHaveCorrectType)}",
				"\tconstant result = value",
				"\tresult is " + nameof(ValueMustHaveCorrectType),
				"\tresult")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(
			((Body)program.Methods[0].GetBodyAndParseIfNeeded()).FindVariable("value")?.Type,
			Is.EqualTo(program));
	}

	[Test]
	public void CanAccessThePropertiesOfValue()
	{
		var program = new Type(type.Package, new TypeLines(
				nameof(CanAccessThePropertiesOfValue),
				"has logger",
				"has Number",
				"has myMember Text",
				"Dummy(dummy Number) Text",
				"\tlet result = value.myMember",
				"\tresult + \"dummy\"")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.FindVariable("value")?.Type, Is.EqualTo(program));
		Assert.That(body.FindVariable("result")?.Type.Name, Is.EqualTo("Text"));
	}

	[TestCase("ProgramWithHas", "numbers",
		"has numbers",
		"Dummy",
		"\tconstant instanceWithNumbers = ProgramWithHas((1, 2, 3))",
		"\tinstanceWithNumbers is ProgramWithHas")]
	[TestCase("ProgramWithPublicMember",
		"texts", "has Texts",
		"Dummy",
		"\tconstant instanceWithTexts = ProgramWithPublicMember((\"1\", \"2\", \"3\"))",
		"\tinstanceWithTexts is ProgramWithPublicMember")]
	public void ParseConstructorCallWithList(string programName, string expected, params string[] code)
	{
		var program = new Type(type.Package, new TypeLines(
				programName,
				code)).
			ParseMembersAndMethods(new MethodExpressionParser());
		var assignment =
			(Declaration)((Body)program.Methods[0].GetBodyAndParseIfNeeded()).Expressions[0];
		Assert.That(((MethodCall)assignment.Value).Method.Parameters[0].Name, Is.EqualTo(expected));
	}

	[Test]
	public void TypeImplementsGenericTypeWithLength()
	{
		new Type(type.Package,
			new TypeLines("HasLengthImplementation",
				"has HasLength",
				"has boolean",
				"from(boolean)",
				"\tboolean = boolean",
				"Length Number",
				"\tvalue")).ParseMembersAndMethods(new MethodExpressionParser());
		var program = new Type(type.Package,
			new TypeLines(nameof(TypeImplementsGenericTypeWithLength),
				"has logger", //unused member should be removed later when we allow class without members
				"GetLengthSquare(type HasLength) Number",
				"\ttype.Length * type.Length",
				"Dummy",
				"\tconstant countOfFive = HasLengthImplementation(true)",
				"\tconstant lengthSquare = GetLengthSquare(countOfFive)")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(program.Methods[1].GetBodyAndParseIfNeeded().ToString(),
			Is.EqualTo("constant countOfFive = HasLengthImplementation(true)\r\n" +
				"constant lengthSquare = GetLengthSquare(countOfFive)"));
	}

	[Test]
	public void MutableCanUseChildMethods()
	{
		var program = new Type(type.Package,
			new TypeLines(nameof(MutableCanUseChildMethods),
				"has logger",
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
			Throws.InstanceOf<Method.RecursiveCallCausesStackOverflow>());
	}

	[Test]
	public void NestedMethodCall()
	{
		var program = new Type(type.Package,
				new TypeLines(nameof(NestedMethodCall), "has logger", "Run",
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
					"has logger",
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
					"has logger",
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
					"has logger",
					"GetResult(number) Number",
					"\tGetResult(10) is 15",
					"\tlet instance = TypeCannotBeAutoInitialized(number)",
					"\tinstance.AddFiveWithInput")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var body = (Body)consumer.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(((Declaration)body.Expressions[1]).Value.ReturnType.Name, Is.EqualTo("TypeCannotBeAutoInitialized"));
	}
}