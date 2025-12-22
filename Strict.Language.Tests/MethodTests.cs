namespace Strict.Language.Tests;

public sealed class MethodTests
{
	[SetUp]
	public void CreateType()
	{
		type = new Type(TestPackage.Instance, new MockRunTypeLines());
		parser = new MethodExpressionParser();
	}

	private Type type = null!;
	private MethodExpressionParser parser = null!;

	[TearDown]
	public void TearDown() => type.Dispose();

	[Test]
	public void MustMustHaveAValidName() =>
		Assert.That(() => new Method(type, 0, null!, ["5(text)"]),
			Throws.InstanceOf<Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers>());

	[Test]
	public void ReturnTypeMustBeBeLast() =>
		Assert.That(() => new Method(type, 0, null!, ["Texts GetFiles"]),
			Throws.InstanceOf<Context.TypeNotFound>());

	[Test]
	public void InvalidMethodParameters() =>
		Assert.Throws<Method.InvalidMethodParameters>(
			() => new Method(type, 0, null!, ["ab("]));

	[Test]
	public void ParametersMustNotBeEmpty() =>
		Assert.That(() => new Method(type, 0, null!, ["ab()"]),
			Throws.InstanceOf<Method.EmptyParametersMustBeRemoved>());

	[TestCase("from(Text)")]
	[TestCase("from(Number)")]
	[TestCase("from(Start Number, End Number)")]
	[TestCase("from(start Number, End Number)")]
	public void UpperCaseParameterWithNoTypeSpecificationIsNotAllowed(string method) =>
		Assert.That(() => new Method(type, 0, null!, [method]),
			Throws.InstanceOf<Method.ParametersMustStartWithLowerCase>());

	[Test]
	public void ParseDefinition()
	{
		var method = new Method(type, 0, null!, [Run]);
		Assert.That(method.Name, Is.EqualTo(Run));
		Assert.That(method.Parameters, Is.Empty);
		Assert.That(method.ReturnType, Is.EqualTo(type.GetType(Base.None)));
		Assert.That(method.ToString(), Is.EqualTo(Run));
	}

	[Test]
	public void ParseFrom()
	{
		var method = new Method(type, 0, null!, ["from(number)"]);
		Assert.That(method.Name, Is.EqualTo("from"));
		Assert.That(method.Parameters, Has.Count.EqualTo(1), method.Parameters.ToWordList());
		Assert.That(method.Parameters[0].Type, Is.EqualTo(type.GetType("Number")));
		Assert.That(method.ReturnType, Is.EqualTo(type));
	}

	public const string Run = nameof(Run);

	[Test]
	public void ParseWithReturnType()
	{
		var method = new Method(type, 0, null!, NestedMethodLines);
		Assert.That(method.Name, Is.EqualTo("IsBlaFive"));
		Assert.That(method.Parameters, Is.Empty);
		Assert.That(method.ReturnType, Is.EqualTo(type.GetType(Base.Boolean)));
		Assert.That(method.ToString(), Is.EqualTo(NestedMethodLines[0]));
	}

	public static readonly string[] NestedMethodLines =
	[
		"IsBlaFive Boolean",
		ConstantNumber,
		"	if bla is 5",
		"		return true",
		"	false"
	];
	public const string ConstantNumber = "	constant number = 5";
	public const string ConstantOther = "	constant other = 3";
	public const string ConstantErrorMessage = "\tconstant errorMessage = \"some error\"";

	[Test]
	public void TraitMethodBodiesShouldNotBeCalled()
	{
		var appTrait =
			new Type(type.Package, new TypeLines("DummyApp", "Run")).ParseMembersAndMethods(null!);
		Assert.That(() => appTrait.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Method.CannotCallBodyOnTraitMethod>());
	}

	[Test]
	public void AccessValidMethodParametersInMethodBody()
	{
		var method = new Method(type, 0, parser, [
			"Run(variable Text)",
			"\tlet result = variable + \"5\"",
			"\tresult"
		]);
		Assert.That(method.Name, Is.EqualTo(Run));
		Assert.That(method.Parameters, Has.Count.EqualTo(1));
		var body = (Body)method.GetBodyAndParseIfNeeded();
		var binary = (Binary)((Declaration)body.Expressions[0]).Value;
		Assert.That(binary.Instance, Is.InstanceOf<ParameterCall>());
	}

	[TestCase("Run(variable Generic)")]
	[TestCase("Run(generic)")]
	[TestCase("Run(number, input Generic, generic)")]
	[TestCase("Run(number) Generic")]
	public void GenericMethods(string methodHeader) =>
		Assert.That(new Method(type, 0, parser, [methodHeader]).IsGeneric, Is.True);

	[TestCase("Run(text) Number")]
	[TestCase("Run(variable Number, input Text) Boolean")]
	public void NonGenericMethods(string methodHeader) =>
		Assert.That(new Method(type, 0, parser, [methodHeader]).IsGeneric, Is.False);

	[Test]
	public void CloningWithSameParameterType()
	{
		var method = new Method(type, 0, parser, ["Run(variable Text)", "	\"5\""]);
		Assert.That(method.Parameters[0].CloneWithImplementationType(type.GetType(Base.Text)),
			Is.EqualTo(method.Parameters[0]));
	}

	[Test]
	public void SplitTestExpressions()
	{
		var customType = new Type(TestPackage.Instance,
			new TypeLines(nameof(SplitTestExpressions),
				"has logger",
				"AddFive(variable Text) Text",
				"	AddFive(\"5\") is \"55\"",
				"	AddFive(\"6\") is \"65\"",
				"	variable + \"5\"")).ParseMembersAndMethods(parser);
		customType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(customType.Methods[0].Tests.Count, Is.EqualTo(2));
	}

	[Test]
	public void ConditionalExpressionIsNotTest()
	{
		var customType = new Type(TestPackage.Instance,
			new TypeLines(nameof(ConditionalExpressionIsNotTest),
				"has logger",
				"ConditionalExpressionIsNotTest Boolean",
				"	5 is 5 ? true else false")).ParseMembersAndMethods(parser);
		customType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(customType.Methods[0].Tests.Count, Is.EqualTo(0));
	}

	[Test]
	public void MethodParameterWithGenericTypeImplementations()
	{
		var method = new Method(type, 0, parser,
			["Run(iterator Iterator(Text), index Number)", "	\"5\""]);
		Assert.That(method.Parameters[0].Name, Is.EqualTo("iterator"));
		Assert.That(method.Parameters[0].Type, Is.EqualTo(type.GetType("Iterator(Text)")));
		Assert.That(method.Parameters[1].Name, Is.EqualTo("index"));
		Assert.That(method.Parameters[1].Type, Is.EqualTo(type.GetType(Base.Number)));
	}

	[Test]
	public void MethodParameterWithDefaultValue()
	{
		var method = new Method(type, 0, parser, ["Run(input = \"Hello\")", "	\"5\""]);
		Assert.That(method.Parameters[0].DefaultValue, Is.EqualTo(new Text(type, "Hello")));
	}

	[Test]
	public void ImmutableMethodParameterValueCannotBeChanged()
	{
		var method = new Method(type, 0, parser, ["Run(input = \"Hello\")", "	input = \"Hi\""]);
		Assert.That(() => method.GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Body.ValueIsNotMutableAndCannotBeChanged>());
	}

	[Test]
	public void ImmutableMethodVariablesCannotBeChanged()
	{
		var method = new Method(type, 0, parser,
			["Run", "	constant random = \"Hi\"", "	random = \"Ho\""]);
		Assert.That(() => method.GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Body.ValueIsNotMutableAndCannotBeChanged>());
	}

	[Test]
	public void ValueTypeNotMatchingWithAssignmentType() =>
		Assert.That(
			() => new Method(type, 0, parser, ["Run(mutable input = 0)", "	input = \"5\""]).
				GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<MutableReassignment.ValueTypeNotMatchingWithAssignmentType>());

	[Test]
	public void MissingParameterDefaultValue() =>
		Assert.That(
			() => new Method(type, 0, parser, ["Run(input =)", "	5"]),
			Throws.InstanceOf<Method.MissingParameterDefaultValue>());

	[Test]
	public void ParameterWithTypeNameAndInitializerIsForbidden() =>
		Assert.That(
			() => new Method(type, 0, parser, ["Run(input Number = 5)", "	5"]),
			Throws.InstanceOf<NamedType.AssignmentWithInitializerTypeShouldNotHaveNameWithType>());

	[Test]
	public void MethodMustHaveAtLeastOneTest() =>
		Assert.That(
			() =>
			{
				using var mockType = new Type(new Package(nameof(MethodMustHaveAtLeastOneTest)),
					new MockRunTypeLines());
				return new Method(mockType, 0, parser, ["NoTestMethod Number", "	5"]).
					GetBodyAndParseIfNeeded();
			}, //ncrunch: no coverage
			Throws.InstanceOf<Method.MethodMustHaveAtLeastOneTest>());

	[Test]
	public void MethodWithTestsAreAllowed()
	{
		using var methodWithTestsType = new Type(
			new Package(TestPackage.Instance, nameof(MethodWithTestsAreAllowed)),
			new TypeLines(nameof(MethodWithTestsAreAllowed), "has logger",
				"MethodWithTestsAreAllowed Number", "\tMethodWithTestsAreAllowed is 5", "\t5"));
		methodWithTestsType.ParseMembersAndMethods(parser);
		Assert.That(() => methodWithTestsType.Methods[0].GetBodyAndParseIfNeeded(), Throws.Nothing);
	}

	[Test]
	public void ParseMethodWithMultipleReturnType()
	{
		using var multipleReturnTypeMethod = new Type(
			new Package(TestPackage.Instance, nameof(ParseMethodWithMultipleReturnType)), new TypeLines(
				"Processor",
			// @formatter:off
			"has progress Number",
			"IsJobDone Boolean or Text",
			"\tProcessor(100).IsJobDone is true",
			"\tProcessor(78).IsJobDone is false",
			"\tProcessor(0).IsJobDone is \"Work not started yet\"",
			"\tif progress is 100",
			"\t\treturn true",
			"\tif progress > 0",
			"\t\treturn false",
			"\t\"Work not started yet\""));
			// @formatter:on
		multipleReturnTypeMethod.ParseMembersAndMethods(parser);
		Assert.That(() => multipleReturnTypeMethod.Methods[0].GetBodyAndParseIfNeeded(), Throws.Nothing);
		Assert.That(multipleReturnTypeMethod.Methods[0].ReturnType, Is.InstanceOf<OneOfType>());
		Assert.That(multipleReturnTypeMethod.Methods[0].ReturnType.Name, Is.EqualTo("BooleanOrText"));
	}

	[Test]
	public void ParseMethodWithParametersAndMultipleReturnType()
	{
		using var multipleReturnTypeMethod = new Type(
			new Package(TestPackage.Instance, nameof(ParseMethodWithParametersAndMultipleReturnType)),
			new TypeLines("Processor",
			// @formatter:off
			"has progress Number",
			"IsJobDone(number, text) Boolean or Text",
			"\tProcessor(100).IsJobDone(1, \"hi\") is true",
			"\tProcessor(78).IsJobDone(1, \"hi\") is false",
			"\tProcessor(0).IsJobDone(1, \"hi\") is \"Work not started yet\"",
			"\tif progress is 100",
			"\t\treturn true",
			"\tif progress > 0",
			"\t\treturn false",
			"\t\"Work not started yet\""));
			// @formatter:on
		multipleReturnTypeMethod.ParseMembersAndMethods(parser);
		Assert.That(() => multipleReturnTypeMethod.Methods[0].GetBodyAndParseIfNeeded(), Throws.Nothing);
	}

	[Test]
	public void MethodCallWithMultipleReturnTypes()
	{
		using var multipleReturnTypeMethod = new Type(
			new Package(TestPackage.Instance, nameof(MethodCallWithMultipleReturnTypes)), new TypeLines(
				"Processor",
			// @formatter:off
			"has progress Number",
			"IsJobDone Boolean or Text",
			"\tProcessor(100).IsJobDone is true",
			"\tProcessor(78).IsJobDone is false",
			"\tProcessor(0).IsJobDone is \"Work not started yet\"",
			"\tif progress is 100",
			"\t\treturn true",
			"\tif progress > 0",
			"\t\treturn false",
			"\t\"Work not started yet\"",
			"Run",
			"\tconstant result = IsJobDone",
			"\tif result",
			"\t\t\"Successfully processed\"",
			"\telse if result is \"Work not started yet\"",
			"\t\treturn Error",
			"\telse",
			"\t\t\"Job is in progress\""));
			// @formatter:on
		multipleReturnTypeMethod.ParseMembersAndMethods(parser);
		Assert.That(() => multipleReturnTypeMethod.Methods[0].GetBodyAndParseIfNeeded(), Throws.Nothing);
		Assert.That(() => multipleReturnTypeMethod.Methods[1].GetBodyAndParseIfNeeded(), Throws.Nothing);
	}

	[Test]
	public void DeclarationIsNeverUsedAndMustBeRemoved()
	{
		using var typeToCheck = new Type(TestPackage.Instance,
			new TypeLines(nameof(DeclarationIsNeverUsedAndMustBeRemoved), "has logger",
				"MethodWithTestsAreAllowed Number", "\tconstant unused = 5"));
		typeToCheck.ParseMembersAndMethods(parser);
		Assert.That(() => typeToCheck.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Method.DeclarationIsNeverUsedAndMustBeRemoved>());
	}

	[Test]
	public void MutableUsesConstantValue()
	{
		using var typeToCheck = new Type(TestPackage.Instance,
			new TypeLines(nameof(MutableUsesConstantValue), "has logger",
				"MethodWithTestsAreAllowed Number", "\tmutable unused = 5", "\t6"));
		typeToCheck.ParseMembersAndMethods(parser);
		Assert.That(() => typeToCheck.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Method.MutableUsesConstantValue>());
	}

	[Test]
	public void GetVariableUsageCount() =>
		Assert.That(
			TestPackage.Instance.GetType(Base.Character).AvailableMethods["to"][0].
				GetVariableUsageCount("notANumber"), Is.EqualTo(3));
}