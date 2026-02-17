using Strict.Language.Tests;

namespace Strict.Expressions.Tests;

// ReSharper disable once TestFileNameWarning
public sealed class ListAdvancedTests : TestExpressions
{
	[SetUp]
	public void CreateParser() => parser = new MethodExpressionParser();

	private MethodExpressionParser parser = null!;

	[Test]
	public void ListPrefixIsNotAllowed() =>
		Assert.That(
			() =>
			{
				using var _ = new Type(type.Package,
						new TypeLines(nameof(ListPrefixIsNotAllowed), "has listOne Numbers")).
					ParseMembersAndMethods(parser);
			}, //ncrunch: no coverage
			Throws.InnerException.
				InstanceOf<Context.ListPrefixIsNotAllowedUseImplementationTypeNameInPlural>());

	[Test]
	public void ListGenericLengthAddition()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(ListGenericLengthAddition), "has ones Numbers", "has twos Numbers",
				"AddListLength Number", "\tones.Length + twos.Length")).ParseMembersAndMethods(parser);
		Assert.That(program.Members[0].Name, Is.EqualTo("ones"));
		var numbersListType = type.GetType(Base.List).
			GetGenericImplementation(type.GetType(Base.Number));
		Assert.That(program.Members[0].Type, Is.EqualTo(numbersListType));
		Assert.That(program.Members[1].Type, Is.EqualTo(numbersListType));
	}

	[Test]
	public void ListAdditionWithGeneric()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(ListAdditionWithGeneric), "has elements Numbers",
				"Add(other Numbers) List", "\telements + other.elements")).ParseMembersAndMethods(parser);
		Assert.That(program.Members[0].Name, Is.EqualTo("elements"));
		Assert.That(program.Methods[0].ReturnType.Name, Is.EqualTo("List"));
		Assert.That(program.Members[0].Type.IsIterator, Is.True);
	}

	[TestCase("Add(input Number) Numbers", "NumbersCompatibleWithCount")]
	[TestCase("Add(input Character) Numbers", "NumbersCompatibleWithCharacter")]
	public void NumbersCompatibleWithImplementedTypes(string code, string testName)
	{
		using var program = new Type(type.Package,
				new TypeLines(testName, "has logger", code, "\t(1, 2, 3, input)")).
			ParseMembersAndMethods(parser);
		Assert.That(program.Methods[0].GetBodyAndParseIfNeeded().ReturnType,
			Is.EqualTo(program.GetListImplementationType(type.GetType(Base.Number))));
	}

	[Test]
	public void NotOperatorInAssignment()
	{
		using var typeWithAssignment = new Type(type.Package,
			new TypeLines(nameof(NotOperatorInAssignment), "has numbers", "NotOperator",
				"\tconstant result = not true", "\tresult is false")).ParseMembersAndMethods(parser);
		var assignment = ((Body)typeWithAssignment.Methods[0].GetBodyAndParseIfNeeded()).Expressions[0];
		Assert.That(assignment.ToString(), Is.EqualTo("constant result = not true"));
	}

	[Test]
	public void UnknownExpressionForArgumentInList() =>
		Assert.That(
			() =>
			{
				using var dummy = new Type(type.Package,
						new TypeLines(nameof(UnknownExpressionForArgumentInList), "has logger",
							"UnknownExpression", "\tconstant result = ((1, 2), 9gf5)")).
					ParseMembersAndMethods(parser);
				dummy.Methods[0].GetBodyAndParseIfNeeded();
			}, //ncrunch: no coverage
			Throws.InstanceOf<UnknownExpressionForArgument>());

	[Test]
	public void AccessListElementsByIndex()
	{
		using var typeWithAccess = new Type(type.Package,
			new TypeLines(nameof(AccessListElementsByIndex), "has numbers",
				"AccessZeroIndexElement Number", "\tnumbers(0)")).ParseMembersAndMethods(parser);
		var expression = typeWithAccess.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(expression.ToString(), Is.EqualTo("numbers(0)"));
	}

	[Test]
	public void AllowMutableListWithEmptyExpressions()
	{
		using var typeWithMutableList = new Type(type.Package,
			new TypeLines(nameof(AllowMutableListWithEmptyExpressions), "has numbers",
				"CreateMutableList Numbers", "\tmutable result = Numbers", "\tfor numbers",
				"\t\tresult = result - value", "\tresult")).ParseMembersAndMethods(parser);
		var expression = (Body)typeWithMutableList.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(expression.Expressions[0].ToString(),
			Is.EqualTo("mutable result = List(Number)"));
		Assert.That(((Declaration)expression.Expressions[0]).Value.ReturnType.FullName,
			Is.EqualTo("TestPackage.List(Number)"));
	}

	[Test]
	public void CreateMemberWithMutableListType()
	{
		using var mutableTextsType = new Type(type.Package,
				new TypeLines(nameof(CreateMemberWithMutableListType), "mutable mutableTexts Texts",
					"AddFiveToMutableList Texts", "\tmutableTexts = mutableTexts + \"5\"")).
			ParseMembersAndMethods(parser);
		var expression = (MutableReassignment)mutableTextsType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(((Binary)expression.Value).ToString(), Is.EqualTo("mutableTexts + \"5\""));
	}

	[Test]
	public void OnlyListTypeIsAllowedAsMutableExpressionArgument() =>
		Assert.That(
			() =>
			{
				using var dummy = new Type(type.Package,
					new TypeLines(nameof(OnlyListTypeIsAllowedAsMutableExpressionArgument),
						"has unused Logger", "MutableWithNumber Number",
						"\tconstant result = Mutable(Number)", "\tresult + 1")).ParseMembersAndMethods(parser);
				dummy.Methods[0].GetBodyAndParseIfNeeded();
			}, //ncrunch: no coverage
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.GenericTypesCannotBeUsedDirectlyUseImplementation>());

	[Test]
	public void CheckIfInvalidArgumentIsNotMethodOrListCall() =>
		Assert.That(
			() =>
			{
				using var dummy = new Type(type.Package,
					new TypeLines(nameof(CheckIfInvalidArgumentIsNotMethodOrListCall), "has booleans",
						"AccessZeroIndexElement Boolean", "\tlet firstValue = booleans(0)",
						"\tfirstValue(0)")).ParseMembersAndMethods(parser);
				dummy.Methods[0].GetBodyAndParseIfNeeded();
			}, //ncrunch: no coverage
			Throws.InstanceOf<InvalidArgumentItIsNotMethodOrListCall>());

	[Test]
	public void MultiLineListsAllowedOnlyIfLengthIsMoreThanHundred() =>
		Assert.That(() =>
			{
				using var dummy = new Type(type.Package, new TypeLines(
						nameof(MultiLineListsAllowedOnlyIfLengthIsMoreThanHundred),
					// @formatter:off
					"has logger",
					"Run",
					"\tconstant result = (1,",
					"\t2,",
					"\t3,",
					"\t4,",
					"\t5,",
					"\t6,",
					"\t7)",
					"ExtraMethodNotCalled",
					"\tsomething"))
					// @formatter:on
					.ParseMembersAndMethods(parser);
			}, //ncrunch: no coverage
			Throws.InstanceOf<TypeParser.MultiLineExpressionsAllowedOnlyWhenLengthIsMoreThanHundred>().
				With.Message.
				Contains("Current length: 40, Minimum Length for Multi line expressions: 100"));

	[Test]
	public void UnterminatedMultiLineListFound() =>
		Assert.That(() =>
			{
				using var dummy = new Type(type.Package, new TypeLines(nameof(UnterminatedMultiLineListFound),
					// @formatter:off
					"has logger",
					"Run",
					"\tconstant result = (1,",
					"\t2,",
					"\t3,",
					"\t4,",
					"ExtraMethodNotCalled",
					"\tsomething"))
					// @formatter:on
					.ParseMembersAndMethods(parser);
			}, //ncrunch: no coverage
			Throws.InstanceOf<TypeParser.UnterminatedMultiLineListFound>().With.Message.
				StartWith("\tconstant result = (1, 2, 3, 4,"));

	// @formatter:off
	[TestCase("ParseMultiLineExpressionWithNumbers", "(numbers(0),\n\tnumbers(1),\n\tnumbers(2),\n\tnumbers(3),\n\tnumbers(4),\n\tnumbers(5)," +
		"\n\tnumbers(6),\n\tanotherNumbers(0),\n\tanotherNumbers(1),\n\tanotherNumbers(2))", "has numbers",
		"has anotherNumbers Numbers",
		"Run Numbers",
		"\t(numbers(0),",
		"\tnumbers(1),",
		"\tnumbers(2),",
		"\tnumbers(3),",
		"\tnumbers(4),",
		"\tnumbers(5),",
		"\tnumbers(6),",
		"\tanotherNumbers(0),",
		"\tanotherNumbers(1),",
		"\tanotherNumbers(2))",
		"ExtraMethodNotCalled",
		"\tsomething")]
	[TestCase("ParseMultiLineExpressionWithText", "(\"somethingsomethingsomething + 5\","+
		"\n\t\"somethingsomethingsomethingsomething\","+
		"\n\t\"somethingsomethingsomethingsomething\","+
		"\n\t\"somethingsomethingsomethingsomething\","+
		"\n\t\"somethingsomethingsomethingsomething\","+
		"\n\t\"somethingsomethingsomethingsomething\","+
		"\n\t\"somethingsomethingsomethingsomething\","+
		"\n\t\"somethingsomethingsomethingsomething\","+
		"\n\t\"somethingsomethingsomethingsomething\","+
		"\n\t\"somethingsomethingsomethingsomething\","+
		"\n\t\"somethingsomethingsomethingsomething\")",
		"has logger",
		"Run Numbers",
		"\t(\"somethingsomethingsomething + 5\",",
		"\t\"somethingsomethingsomethingsomething\",",
		"\t\"somethingsomethingsomethingsomething\",",
		"\t\"somethingsomethingsomethingsomething\",",
		"\t\"somethingsomethingsomethingsomething\",",
		"\t\"somethingsomethingsomethingsomething\",",
		"\t\"somethingsomethingsomethingsomething\",",
		"\t\"somethingsomethingsomethingsomething\",",
		"\t\"somethingsomethingsomethingsomething\",",
		"\t\"somethingsomethingsomethingsomething\",",
		"\t\"somethingsomethingsomethingsomething\")",
		"ExtraMethodNotCalled",
		"\tsomething")]
	// @formatter:on
	public void ParseMultiLineExpressionAndPrintSameAsInput(string testName, string expected,
		params string[] code)
	{
		using var program =
			new Type(type.Package, new TypeLines(testName, code)).ParseMembersAndMethods(parser);
		var expression = program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(expression, Is.InstanceOf<List>());
		Assert.That(expression.ToString(), Is.EqualTo(expected));
		Assert.That(program.Methods[1].lines.Count, Is.EqualTo(2));
	}

	[Test]
	public void MergeFromConstructorParametersIntoListIfMemberMatches()
	{
		using var program = new Type(type.Package, new TypeLines(
			// @formatter:off
			"Vector2",
			"has numbers with Length is 2",
			"constant One = Vector2(1, 1)",
			"Length Number",
			"\tVector2.Length is 0",
			"\tVector2(3, 4).Length is 5",
			"\t(X * X + Y * Y).SquareRoot")).ParseMembersAndMethods(parser);
		Assert.That(program.Members[1].Name, Is.EqualTo("One"));
		Assert.That(program.Members[1].Type.ToString(), Is.EqualTo("TestPackage.Vector2"));
	}

	[Test]
	public void FromConstructorCannotBeCreatedWhenFirstMemberIsNotMatched() =>
		Assert.That(() =>
			{
				using var _ = new Type(type.Package, new TypeLines(
					"CannotCreateFromConstructor",
					"constant One = CannotCreateFromConstructor(1)",
					"has numbers with Length is 2",
					"Length Number",
					"\t(X * X + Y * Y).SquareRoot")).ParseMembersAndMethods(parser);
					// @formatter:on
			}, //ncrunch: no coverage
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>());

	[TestCase("numbers", "1, 2", "List(Number)")]
	[TestCase("booleans", "true, false", "List(Boolean)")]
	[TestCase("texts", "\"Hi\", \"Hello\"", "List(Text)")]
	public void AutoParseArgumentAsListIfMatchingWithMethodParameter(string parameter,
		string arguments, string expectedList)
	{
		// @formatter:off
		using var typeWithTestMethods = new Type(type.Package,
			new TypeLines("ListArgumentsCanBeAutoParsed" + parameter,
				"has logger",
				$"CheckInputLengthAndGetResult({parameter}) Number",
				"\tif numbers.Length is 2",
				"\t\treturn 2",
				"\t0",
				"InvokeTestMethod(numbers) Number",
				"\tif numbers.Length is 2",
				"\t\treturn 2",
				$"\tCheckInputLengthAndGetResult({arguments})")).ParseMembersAndMethods(parser);
		// @formatter:on
		var body = (Body)typeWithTestMethods.Methods[1].GetBodyAndParseIfNeeded();
		Assert.That(body.Expressions[1], Is.InstanceOf<MethodCall>());
		var argumentExpression = ((MethodCall)body.Expressions[1]).Arguments[0];
		Assert.That(argumentExpression, Is.InstanceOf<List>());
		Assert.That(argumentExpression.ReturnType.ToString(),
			Is.EqualTo("TestPackage." + expectedList));
	}

	[Test]
	public void CreateMutableListWithMutableExpressions()
	{
		using var program = new Type(TestPackage.Instance, new TypeLines(
					// @formatter:off
					nameof(CreateMutableListWithMutableExpressions),
					"has logger",
					"Add(element Number) Mutable(List)",
					"\tmutable someList = List(Mutable(Number))",
					"\tsomeList.Add(1)")).
			ParseMembersAndMethods(parser);
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.Expressions[0].ToString(), Is.EqualTo("mutable someList = List(Mutable(Number))"));
	}

	[Test]
	public void ChangeValueInsideMutableListWithMutableExpressions()
	{
		using var program = new Type(TestPackage.Instance,
			new TypeLines(nameof(ChangeValueInsideMutableListWithMutableExpressions),
				"has logger",
				"Update(element Number) List(Mutable(Number))",
				"\tmutable someList = List(Mutable(Number))",
				"\tsomeList.Add(1)",
				"\tsomeList(0) = 5",
				"\tsomeList")).
			ParseMembersAndMethods(parser);
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(((VariableCall)((ListCall)((MutableReassignment)body.Expressions[2]).Target).List).
			Variable.Name, Is.EqualTo("someList"));
	}

	[Test]
	public void NegativeIndexIsNeverAllowed()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(NegativeIndexIsNeverAllowed),
				"has logger",
				"UpdateNotExistingElement(element Number) Number",
				"\tmutable someList = List(Mutable(Number))",
				"\tsomeList(-1) = 1")).
			ParseMembersAndMethods(parser);
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(), Throws.InstanceOf<ListCall.NegativeIndexIsNeverAllowed>());
	}

	[Test]
	public void UpdateListExpressionValuesByIndex()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(UpdateListExpressionValuesByIndex),
				"has logger",
				"UpdateListValue(element Number) Number",
				"\tmutable someList = (9, 8, 7)",
				"\tsomeList(0) = 5")).
			ParseMembersAndMethods(parser);
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(((ListCall)((MutableReassignment)body.Expressions[1]).Target).Index.ToString(),
			Is.EqualTo("0"));
	}

	[Test]
	public void IndexAboveConstantListLength()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(IndexAboveConstantListLength),
				"has logger",
				"UpdateNotExistingElement(element Number) Number",
				"\tmutable someList = (9, 8, 7)",
				"\tsomeList(3) = 5")).
			ParseMembersAndMethods(parser);
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(), Throws.InstanceOf<ListCall.IndexAboveConstantListLength>());
	}

	[Test]
	public void IndexViolatesListConstraint()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(IndexViolatesListConstraint),
				"has numbers with Length is 2",
				"from",
				"\tnumbers(3) = 5")).
			ParseMembersAndMethods(parser);
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(), Throws.InstanceOf<ListCall.IndexViolatesListConstraint>());
	}

	[Test]
	public void IndexCheckEvenWorkWhenIndexIsConstant()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(IndexCheckEvenWorkWhenIndexIsConstant),
				"has numbers with Length is 2",
				"from",
				"\tconstant notValid = 5",
				"\tnumbers(notValid) = 5")).
			ParseMembersAndMethods(parser);
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(), Throws.InstanceOf<ListCall.IndexViolatesListConstraint>());
	}

	[Test]
	public void IndexCheckAlsoWorksForMemberCalls()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(IndexCheckAlsoWorksForMemberCalls),
				"has numbers with Length is 2",
				"constant invalidIndex = 3",
				"from",
				"\tnumbers(invalidIndex) = 5")).
			ParseMembersAndMethods(parser);
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(), Throws.InstanceOf<ListCall.IndexViolatesListConstraint>());
	}

	[Test]
	public void IndexCannotBeCheckedOnADynamicCall()
	{
		using var program = new Type(type.Package,
			new TypeLines(nameof(IndexCannotBeCheckedOnADynamicCall),
				"has numbers with Length is 2",
				"from(number)",
				"\tnumbers(number) = 5")).
			ParseMembersAndMethods(parser);
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(), Throws.Nothing);
	}
}