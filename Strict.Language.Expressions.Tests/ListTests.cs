using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

// ReSharper disable once ClassTooBig
public sealed class ListTests : TestExpressions
{
	[SetUp]
	public void CreateParser() => parser = new MethodExpressionParser();

	private MethodExpressionParser parser = null!;

	[Test]
	public void EmptyListNotAllowed() =>
		Assert.That(() => ParseExpression("()"), Throws.InstanceOf<List.EmptyListNotAllowed>()!);

	[TestCase("(1)", 1)]
	[TestCase("(5.23)", 5.23)]
	public void ParseSingleElementLists(string input, double expectedElement) =>
		ParseAndCheckOutputMatchesInput(input,
			new List(new Body(method), new List<Expression> { new Number(method, expectedElement) }));

	[TestCase("(\"1\", 2)")]
	[TestCase("(1, 2.5, \"5\")")]
	[TestCase("(true, 2, \"5\")")]
	public void ListWithMismatchingElementTypes(string input) =>
		Assert.That(() => ParseExpression(input),
			Throws.InstanceOf<List.ListElementsMustHaveMatchingType>());

	[TestCase("(true, false, true, false)", "true, false, true, false")]
	[TestCase("(\"1\", \"2\", \"3\", \"4\", \"5\")", "\"1\", \"2\", \"3\", \"4\", \"5\"")]
	public void ParseLists(string input, string expected) =>
		ParseAndCheckOutputMatchesInput(input,
			new List(new Body(method), GetListExpressions(expected.Split(", "))));

	private List<Expression> GetListExpressions(IEnumerable<string> elements)
	{
		var expressions = new List<Expression>();
		var body = new Body(method);
		foreach (var elementWithSpace in elements)
			AddElementExpression(expressions, elementWithSpace.TrimStart(), body);
		return expressions;
	}

	private void AddElementExpression(ICollection<Expression> expressions, string element, Body body)
	{
		if (element.Length > 3)
			new PhraseTokenizer(element).ProcessEachToken(tokenRange =>
			{
				if (element[tokenRange.Start.Value] != ',')
					expressions.Add(method.ParseExpression(body, element[tokenRange]));
			});
		else
			expressions.Add(method.ParseExpression(body, element));
	}

	[TestCase("(5, -5)")]
	[TestCase("(5, 2 + 5)")]
	public void ListElementsWithMatchingParentType(string code) =>
		Assert.That(ParseExpression(code), Is.InstanceOf<List>());

	[Test]
	public void ParseLists() =>
		ParseAndCheckOutputMatchesInput("((1, 3), (2, 4))",
			new List(new Body(method), GetListExpressions(new[] { "(1, 3)", "(2, 4)" })));

	[TestCase("(1, 2, 3, 4, 5) + \"4\"")]
	[TestCase("(1, 2, 3, 4, 5) + (\"hello\")")]
	[TestCase("(1, 2, 3, 4, 5) + \"hello\" + 4")]
	[TestCase("(1, 2, 3, 4, 5) + (\"hello\") + 4")]
	public void MismatchingTypeFound(string input) =>
		Assert.That(() => ParseExpression(input),
			Throws.InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>()!);

	[TestCase("(1, 2, 3) * (1, 2)")]
	[TestCase("(1, 2, 3) * (1, 2, 3, 4)")]
	public void ListsHaveDifferentDimensionsIsNotAllowed(string input) =>
		Assert.That(() => ParseExpression(input),
			Throws.InstanceOf<Binary.ListsHaveDifferentDimensions>()!);

	[TestCase("(1, 2, 3, 4, 5) + (6, 7, 8)", "1, 2, 3, 4, 5", "+", "6, 7, 8")]
	[TestCase("(1, 2, 3, 4, 5) - (6, 7, 8)", "1, 2, 3, 4, 5", "-", "6, 7, 8")]
	[TestCase("(1, 2, 3, 4, 5) is (1, 2, 3, 4, 5)", "1, 2, 3, 4, 5", "is", "1, 2, 3, 4, 5")]
	[TestCase("(1, 2, 3, 4, 5) * (1, 2, 3, 4, 5)", "1, 2, 3, 4, 5", "*", "1, 2, 3, 4, 5")]
	public void ParseListsBinaryOperation(string input, params string[] expected) =>
		ParseAndCheckOutputMatchesInput(input,
			CreateBinary(new List(new Body(method), GetListExpressions(expected[0].Split(","))),
				expected[1], new List(new Body(method), GetListExpressions(expected[2].Split(",")))));

	[TestCase("(1, 2, 3, 4, 5) + 4", 4, "1, 2, 3, 4, 5", "+")]
	[TestCase("(1, 2, 3, 4, 5) - 4", 4, "1, 2, 3, 4, 5", "-")]
	public void
		ParseListsWithNumber(string input, double expectedRight, params string[] expected) =>
		ParseAndCheckOutputMatchesInput(input,
			CreateBinary(new List(new Body(method), GetListExpressions(expected[0].Split(","))),
				expected[1], new Number(method, expectedRight)));

	[TestCase("(\"1\", \"2\", \"3\", \"4\") + \"5\"", "\"1\", \"2\", \"3\", \"4\"", "+", "5")]
	public void ParseListsWithString(string input, params string[] expected) =>
		ParseAndCheckOutputMatchesInput(input,
			CreateBinary(new List(new Body(method), GetListExpressions(expected[0].Split(","))),
				expected[1], new Text(method, expected[2])));

	[Test]
	public void LeftTypeShouldNotBeChanged()
	{
		const string Code = "(\"1\", \"2\", \"3\", \"4\") + 5";
		var parsedExpression = ParseExpression(Code);
		Assert.That(parsedExpression, Is.InstanceOf<Binary>()!);
		Assert.That(parsedExpression.ToString(), Is.EqualTo(Code));
		Assert.That(((Binary)parsedExpression).ReturnType,
			Is.EqualTo(type.GetType(Base.Text.MakeItPlural())));
	}

	[Test]
	public void LeftTypeShouldNotBeChangedUnlessRightIsList() =>
		Assert.That(() => ParseExpression("5 + (\"1\", \"2\", \"3\", \"4\")"),
			Throws.InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>());

	[TestCase("Number(5)")]
	[TestCase("Text(\"Hi\")")]
	[TestCase("Boolean(true)")]
	[TestCase("List((1, 2))")]
	[TestCase("List((\"Hi\", \"There\"))")]
	public void ConstructorForSameTypeArgumentIsNotAllowed(string code) =>
		Assert.That(() => ParseExpression(code),
			Throws.InstanceOf<MethodCall.ConstructorForSameTypeArgumentIsNotAllowed>());

	[Test]
	public void MemberWithListPrefixInTypeIsNotAllowed() =>
		Assert.That(
			() => new Type(type.Package,
					new TypeLines(nameof(MemberWithListPrefixInTypeIsNotAllowed),
						"has elements List(Number)", "has something Number",
						"AddSomethingWithListLength Number", "\telements.Length + something")).
				ParseMembersAndMethods(parser),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.InstanceOf<NamedType.ListPrefixIsNotAllowedUseImplementationTypeNameInPlural>()!.With.
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

	[TestCase("(1, 2)", Base.Number)]
	[TestCase("(\"1\", \"2\")", Base.Text)]
	[TestCase("(true, false)", Base.Boolean)]
	public void ListShouldHaveCorrectImplementationReturnType(string code, string expectedType) =>
		Assert.That(ParseExpression(code).ReturnType,
			Is.EqualTo(type.GetListImplementationType(type.GetType(expectedType))));

	[Test]
	public void ParseMultipleListInBinary() =>
		ParseAndCheckOutputMatchesInput("(1, 2, 3, 4, 5) + (1) + 4",
			CreateBinary(new List(new Body(method), GetListExpressions("1, 2, 3, 4, 5".Split(", "))),
				BinaryOperator.Plus,
				CreateBinary(new List(new Body(method), GetListExpressions("1".Split(", "))),
					BinaryOperator.Plus, new Number(method, 4))));

	[Test]
	public void ParseNestedLists() =>
		ParseAndCheckOutputMatchesInput("((1, 2, 3) + (3, 4), (4))",
			new List(new Body(method),
				new List<Expression>
				{
					CreateBinary(new List(new Body(method), GetListExpressions("1, 2, 3".Split(", "))),
						BinaryOperator.Plus,
						new List(new Body(method), GetListExpressions("3, 4".Split(", ")))),
					new List(new Body(method), GetListExpressions("4".Split(", ")))
				}));

	[Test]
	public void ParseComplexLists() =>
		ParseAndCheckOutputMatchesInput(
			"((\"Hello, World\", \"Yoyo (it is my secret + 1)\"), (\"4\")) + (\"7\")",
			CreateBinary(
				new List(new Body(method),
					GetListExpressions(new[]
					{
						"(\"Hello, World\", \"Yoyo (it is my secret + 1)\"), (\"4\")"
					})), BinaryOperator.Plus,
				new List(new Body(method), new List<Expression> { new Text(method, "7") })));

	[Test]
	public void ListGenericLengthAddition()
	{
		var program = new Type(type.Package,
			new TypeLines(nameof(ListGenericLengthAddition), "has listOne Numbers", "has listTwo Numbers", "AddListLength Number", "\tlistOne.Length + listTwo.Length")).ParseMembersAndMethods(parser);
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
			new TypeLines(nameof(ListAdditionWithGeneric), "has elements Numbers", "Add(other Numbers) List", "\telements + other.elements")).ParseMembersAndMethods(parser);
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
					"\tconstant result = ((not true))")).ParseMembersAndMethods(parser).Methods[0].
			GetBodyAndParseIfNeeded();
		Assert.That(assignment.ToString(), Is.EqualTo("constant result = (not true)"));
	}

	[Test]
	public void UnknownExpressionForArgumentInList() =>
		Assert.That(() => new Type(type.Package,
				new TypeLines(nameof(UnknownExpressionForArgumentInList), "has log", "UnknownExpression",
					"\tconstant result = ((1, 2), 9gfhy5)")).ParseMembersAndMethods(parser).Methods[0].
			GetBodyAndParseIfNeeded(), Throws.InstanceOf<UnknownExpressionForArgument>()!);

	[Test]
	public void AccessListElementsByIndex()
	{
		var expression = new Type(type.Package,
			new TypeLines(nameof(AccessListElementsByIndex), "has numbers", "AccessZeroIndexElement Number", "\tnumbers(0)")).ParseMembersAndMethods(parser).Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(expression.ToString(), Is.EqualTo("numbers(0)"));
	}

	[Test]
	public void AllowMutableListWithEmptyExpressions()
	{
		var expression = (Body)new Type(type.Package,
			new TypeLines(nameof(AllowMutableListWithEmptyExpressions),
				"has numbers",
				"CreateMutableList Numbers",
				"\tmutable result = Numbers",
				"\tfor numbers",
				"\t\tresult = result + (0 - value)",
				"\tresult")).ParseMembersAndMethods(parser).Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(expression.Expressions[0].ToString(), Is.EqualTo("mutable result = Numbers"));
		Assert.That(((ConstantDeclaration)expression.Expressions[0]).Value.ReturnType.FullName,
			Is.EqualTo("TestPackage.Numbers"));
	}

	[Test]
	public void CreateMemberWithMutableListType()
	{
		var mutableTextsType = new Type(type.Package,
			new TypeLines(nameof(CreateMemberWithMutableListType),
				"mutable mutableTexts Texts",
				"AddFiveToMutableList Texts",
				"\tmutableTexts = mutableTexts + 5",
				"\tmutableTexts")).ParseMembersAndMethods(parser);
		var expression = (Body)mutableTextsType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(mutableTextsType.Members[0].Value?.ToString(),
			Is.EqualTo("mutableTexts + 5"));
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
	public void ContainsMethodCallOnNumbersList()
	{
		var ifExpression = ParseExpression(
			"if (1, 2, 3).Contains(2)",
			"\tconstant abc = \"abc\"",
			"\tlog.Write(abc)") as If;
		var numbers =
			(ifExpression?.Condition as MethodCall)?.Instance as List;
		Assert.That(numbers?.ToString(), Is.EqualTo("(1, 2, 3)"));
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

	[Test]
	public void MethodsAndMembersOfListShouldHaveImplementationTypeAsParent()
	{
		var numbers = type.GetListImplementationType(type.GetType(Base.Number));
		Assert.That(numbers.Members[1].ToString(),
			Is.EqualTo("elements TestPackage.Numbers"));
		Assert.That(numbers.Methods[1].Parent.ToString(),
			Is.EqualTo("TestPackage.Numbers"));
	}

	[Test]
	public void MethodBodyShouldBeUpdatedWithImplementationType()
	{
		var texts = type.GetListImplementationType(type.GetType(Base.Text));
		var containsMethod = texts.Methods.FirstOrDefault(m => m.Name == "Contains");
		Assert.That(((Body)containsMethod!.GetBodyAndParseIfNeeded()).Method,
			Is.EqualTo(containsMethod));
	}
}