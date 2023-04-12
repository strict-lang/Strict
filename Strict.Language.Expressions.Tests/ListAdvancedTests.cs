using NUnit.Framework;
using Strict.Language.Tests;

namespace Strict.Language.Expressions.Tests;

public sealed class ListAdvancedTests : TestExpressions
{
	[SetUp]
	public void CreateParser() => parser = new MethodExpressionParser();

	private MethodExpressionParser parser = null!;

	[Test]
	public void MemberWithListPrefixInTypeIsNotAllowed() =>
		Assert.That(
			() => new Type(type.Package,
				new TypeLines(nameof(MemberWithListPrefixInTypeIsNotAllowed), "has elements List(Number)",
					"has something Number", "AddSomethingWithListLength Number",
					"\telements.Length + something")).ParseMembersAndMethods(parser),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<NamedType.ListPrefixIsNotAllowedUseImplementationTypeNameInPlural>()!.With.
				Message.Contains(
					"List should not be used as prefix for List(Number) instead use Numbers"));

	[Test]
	public void MethodParameterWithListPrefixInTypeIsNotAllowed() =>
		Assert.That(
			() => new Type(type.Package,
					new TypeLines(nameof(MethodParameterWithListPrefixInTypeIsNotAllowed), "has log",
						"AddNumberToTexts(input List(Text), number) List(Text)", "\tinput + number")).
				ParseMembersAndMethods(parser),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<NamedType.ListPrefixIsNotAllowedUseImplementationTypeNameInPlural>()!.With.
				Message.Contains("List should not be used as prefix for List(Text) instead use Texts"));

	[Test]
	public void ListGenericLengthAddition()
	{
		var program = new Type(type.Package,
				new TypeLines(nameof(ListGenericLengthAddition), "has listOne Numbers",
					"has listTwo Numbers", "AddListLength Number", "\tlistOne.Length + listTwo.Length")).
			ParseMembersAndMethods(parser);
		Assert.That(program.Members[0].Name, Is.EqualTo("listOne"));
		var numbersListType = type.GetType(Base.List).
			GetGenericImplementation(new List<Type> { type.GetType(Base.Number) });
		Assert.That(program.Members[0].Type, Is.EqualTo(numbersListType));
		Assert.That(program.Members[1].Type, Is.EqualTo(numbersListType));
	}

	[Test]
	public void ListAdditionWithGeneric()
	{
		var program = new Type(type.Package,
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
		var program = new Type(type.Package,
				new TypeLines(testName, "has log", code, "\tconstant result = (1, 2, 3, input)")).
			ParseMembersAndMethods(parser);
		Assert.That(program.Methods[0].GetBodyAndParseIfNeeded().ReturnType,
			Is.EqualTo(program.GetListImplementationType(type.GetType(Base.Number))));
	}

	[Test]
	public void NotOperatorInAssignment()
	{
		var assignment = (ConstantDeclaration)new Type(type.Package,
				new TypeLines(nameof(NotOperatorInAssignment), "has numbers", "NotOperator",
					"\tconstant result = not true")).ParseMembersAndMethods(parser).Methods[0].
			GetBodyAndParseIfNeeded();
		Assert.That(assignment.ToString(), Is.EqualTo("constant result = not true"));
	}

	[Test]
	public void UnknownExpressionForArgumentInList() =>
		Assert.That(
			() => new Type(type.Package,
					new TypeLines(nameof(UnknownExpressionForArgumentInList), "has log",
						"UnknownExpression",
						"\tconstant result = ((1, 2), 9gfhy5)")).ParseMembersAndMethods(parser).Methods[0].
				GetBodyAndParseIfNeeded(), Throws.InstanceOf<UnknownExpressionForArgument>()!);

	[Test]
	public void AccessListElementsByIndex()
	{
		var expression = new Type(type.Package,
				new TypeLines(nameof(AccessListElementsByIndex), "has numbers",
					"AccessZeroIndexElement Number", "\tnumbers(0)")).ParseMembersAndMethods(parser).
			Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(expression.ToString(), Is.EqualTo("numbers(0)"));
	}

	[Test]
	public void AllowMutableListWithEmptyExpressions()
	{
		var expression = (Body)new Type(type.Package,
				new TypeLines(nameof(AllowMutableListWithEmptyExpressions), "has numbers",
					"CreateMutableList Numbers", "\tmutable result = Numbers", "\tfor numbers",
					"\t\tresult = result + (0 - value)", "\tresult")).ParseMembersAndMethods(parser).
			Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(expression.Expressions[0].ToString(), Is.EqualTo("mutable result = Numbers"));
		Assert.That(((ConstantDeclaration)expression.Expressions[0]).Value.ReturnType.FullName,
			Is.EqualTo("TestPackage.Numbers"));
	}

	[Test]
	public void CreateMemberWithMutableListType()
	{
		var mutableTextsType = new Type(type.Package,
				new TypeLines(nameof(CreateMemberWithMutableListType), "mutable mutableTexts Texts",
					"AddFiveToMutableList Texts", "\tmutableTexts = mutableTexts + 5", "\tmutableTexts")).
			ParseMembersAndMethods(parser);
		var expression = (Body)mutableTextsType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(mutableTextsType.Members[0].Value?.ToString(), Is.EqualTo("mutableTexts + 5"));
		Assert.That(((MemberCall)expression.Expressions[0]).Member.Value?.ToString(),
			Is.EqualTo("mutableTexts + 5"));
	}

	[Test]
	public void OnlyListTypeIsAllowedAsMutableExpressionArgument() =>
		Assert.That(
			() => new Type(type.Package,
					new TypeLines(nameof(OnlyListTypeIsAllowedAsMutableExpressionArgument),
						"has unused Log",
						"MutableWithNumber Number", "\tconstant result = Mutable(Number)", "\tresult")).
				ParseMembersAndMethods(parser).Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Type.NoMatchingMethodFound>());

	[Test]
	public void CheckIfInvalidArgumentIsNotMethodOrListCall() =>
		Assert.That(
			() => new Type(type.Package,
					new TypeLines(nameof(CheckIfInvalidArgumentIsNotMethodOrListCall), "has booleans",
						"AccessZeroIndexElement Boolean", "\tconstant firstValue = booleans(0)", "\tfirstValue(0)")).
				ParseMembersAndMethods(parser).Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<InvalidArgumentItIsNotMethodOrListCall>());

	[Test]
	public void MultiLineListsAllowedOnlyIfLengthIsMoreThanHundred() =>
		Assert.That(() => new Type(type.Package, new TypeLines(
					nameof(MultiLineListsAllowedOnlyIfLengthIsMoreThanHundred),
					// @formatter:off
					"has log",
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
				.ParseMembersAndMethods(parser).Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Type.MultiLineExpressionsAllowedOnlyWhenLengthIsMoreThanHundred>().With.Message.
				Contains("Current length: 40, Minimum Length for Multi line expressions: 100"));

	[Test]
	public void UnterminatedMultiLineListFound() =>
		Assert.That(() => new Type(type.Package, new TypeLines(nameof(UnterminatedMultiLineListFound),
					// @formatter:off
					"has log",
					"Run",
					"\tconstant result = (1,",
					"\t2,",
					"\t3,",
					"\t4,",
					"ExtraMethodNotCalled",
					"\tsomething"))
				// @formatter:on
				.ParseMembersAndMethods(parser).Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Type.UnterminatedMultiLineListFound>().With.Message.
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
		"has log",
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
	public void ParseMultiLineExpressionAndPrintSameAsInput(string testName, string expected, params string[] code)
	{
		var program = new Type(type.Package,
				new TypeLines(testName, code)).
			ParseMembersAndMethods(parser);
		var expression = program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(expression, Is.InstanceOf<List>());
		Assert.That(expression.ToString(), Is.EqualTo(expected));
		Assert.That(program.Methods[1].lines.Count, Is.EqualTo(2));
	}

	[Test]
	public void MergeFromConstructorParametersIntoListIfMemberMatches()
	{
		var program = new Type(type.Package,
				new TypeLines(
					// @formatter:off
					"Vector2",
					"has numbers with Length is 2",
					"has One = Vector2(1, 1)",
					"Length Number",
					"\tVector2.Length is 0",
					"\tVector2(3, 4).Length is 5",
					"\t(X * X + Y * Y).SquareRoot")).
			ParseMembersAndMethods(parser);
		Assert.That(program.Members[1].Name, Is.EqualTo("One"));
		Assert.That(program.Members[1].Type.ToString(), Is.EqualTo("TestPackage.Vector2"));
	}

	[Test]
	public void FromConstructorCannotBeCreatedWhenFirstMemberIsNotMatched() =>
		Assert.That(() => new Type(type.Package, new TypeLines(
				"CannotCreateFromConstructor",
				"has One = CannotCreateFromConstructor(1, 1)",
				"has numbers with Length is 2",
				"Length Number",
				"\t(X * X + Y * Y).SquareRoot")).ParseMembersAndMethods(parser),
			Throws.InstanceOf<ParsingFailed>()!.With.InnerException.
				InstanceOf<Type.NoMatchingMethodFound>());
	// @formatter:on

	[TestCase("numbers", "1, 2", "Numbers")]
	[TestCase("booleans", "true, false", "Booleans")]
	[TestCase("texts", "\"Hi\", \"Hello\"", "Texts")]
	public void AutoParseArgumentAsListIfMatchingWithMethodParameter(string parameter, string arguments, string expectedList)
	{
		// @formatter:off
		var typeWithTestMethods = new Type(type.Package,
			new TypeLines("ListArgumentsCanBeAutoParsed" + expectedList,
				"has log",
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
			Is.EqualTo($"TestPackage.{expectedList}"));
	}

	[Test]
	public void CreateMutableListWithMutableExpressions()
	{
		var program = new Type(new TestPackage(),
				new TypeLines(
					// @formatter:off
					nameof(CreateMutableListWithMutableExpressions),
					"has log",
					"Add(element Number) Mutable(List)",
					"\tmutable someList = List(Mutable(Number))",
					"\tsomeList.Add(1)")).
			ParseMembersAndMethods(parser);
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.Expressions[0].ToString(), Is.EqualTo("mutable someList = List(TestPackage.Mutable(TestPackage.Number))"));
	}

	[Test]
	public void ChangeValueInsideMutableListWithMutableExpressions()
	{
		var program = new Type(new TestPackage(),
				new TypeLines(
					// @formatter:off
					nameof(ChangeValueInsideMutableListWithMutableExpressions),
					"has log",
					"Update(element Number) Mutable(List)",
					"\tmutable someList = List(Mutable(Number))",
					"\tsomeList.Add(1)",
					"\tsomeList(0) = 5")).
			ParseMembersAndMethods(parser);
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(((VariableCall)((ListCall)body.Expressions[2]).List).CurrentValue.ToString(), Is.EqualTo("(5)"));
	}

	[Test]
	public void UpdateListExpressionValuesByIndex()
	{
		var program = new Type(type.Package,
				new TypeLines(
					// @formatter:off
					nameof(UpdateListExpressionValuesByIndex),
					"has log",
					"UpdateListValue(element Number) Number",
					"\tmutable someList = (9, 8, 7)",
					"\tsomeList(0) = 5")).
			ParseMembersAndMethods(parser);
		var body = (Body)program.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(((VariableCall)((ListCall)body.Expressions[1]).List).CurrentValue.ToString(), Is.EqualTo("(5, 8, 7)"));
	}

	[Test]
	public void IndexOutOfRangeInListExpressions()
	{
		var program = new Type(type.Package,
				new TypeLines(
					// @formatter:off
					nameof(IndexOutOfRangeInListExpressions),
					"has log",
					"UpdateNotExistingElement(element Number) Number",
					"\tmutable someList = (9, 8, 7)",
					"\tsomeList(3) = 5")).
			ParseMembersAndMethods(parser);
		Assert.That(() => program.Methods[0].GetBodyAndParseIfNeeded(), Throws.InstanceOf<List.IndexOutOfRangeInListExpressions>()!);
	}
}