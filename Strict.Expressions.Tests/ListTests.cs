namespace Strict.Expressions.Tests;

public sealed class ListTests : TestExpressions
{
	[Test]
	public void EmptyListNotAllowed() =>
		Assert.That(() => ParseExpression("()"), Throws.InstanceOf<List.EmptyListNotAllowed>());

	[Test]
	public void CompareList() =>
		Assert.That(new List(new Body(method), GetListExpressions(["\"1\"", "\"2\""])),
			Is.EqualTo(new List(new Body(method), GetListExpressions(["\"1\"", "\"2\""]))));

	[TestCase("(1)", 1)]
	[TestCase("(5.23)", 5.23)]
	public void ParseSingleElementLists(string input, double expectedElement) =>
		ParseAndCheckOutputMatchesInput(input,
			new List(new Body(method), [new Number(method, expectedElement)]));

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

	private void AddElementExpression(ICollection<Expression> expressions, string element,
		Body body)
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
			new List(new Body(method), GetListExpressions(["(1, 3)", "(2, 4)"])));

	[TestCase("(1, 2, 3, 4, 5) + \"4\"")]
	[TestCase("(1, 2, 3, 4, 5) + (\"hello\")")]
	[TestCase("(1, 2, 3, 4, 5) + \"hello\" + 4")]
	[TestCase("(1, 2, 3, 4, 5) + (\"hello\") + 4")]
	public void MismatchingTypeFound(string input) =>
		Assert.That(() => ParseExpression(input),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>());

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
		const string Code = "(\"1\", \"2\", \"3\", \"4\") + \"5\"";
		var parsedExpression = ParseExpression(Code);
		Assert.That(parsedExpression, Is.InstanceOf<Binary>());
		Assert.That(parsedExpression.ToString(), Is.EqualTo(Code));
		Assert.That(((Binary)parsedExpression).ReturnType,
			Is.EqualTo(type.GetType(Base.Text.Pluralize())));
	}

	[Test]
	public void LeftTypeShouldNotBeChangedUnlessRightIsList() =>
		Assert.That(() => ParseExpression("5 + (\"1\", \"2\", \"3\", \"4\")"),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>());

	[TestCase("Number(5)")]
	[TestCase("Text(\"Hi\")")]
	[TestCase("Boolean(true)")]
	[TestCase("List((1, 2))")]
	[TestCase("List((\"Hi\", \"There\"))")]
	public void ConstructorForSameTypeArgumentIsNotAllowed(string code) =>
		Assert.That(() => ParseExpression(code),
			Throws.InstanceOf<MethodCall.ConstructorForSameTypeArgumentIsNotAllowed>());

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
		ParseAndCheckOutputMatchesInput("((1, 2, 3) + (3, 4), (4))", new List(new Body(method), [
			CreateBinary(new List(new Body(method), GetListExpressions("1, 2, 3".Split(", "))),
				BinaryOperator.Plus, new List(new Body(method), GetListExpressions("3, 4".Split(", ")))),
			new List(new Body(method), GetListExpressions("4".Split(", ")))
		]));

	[Test]
	public void ParseComplexLists() =>
		ParseAndCheckOutputMatchesInput(
			"((\"Hello, World\", \"Yoyo (it is my secret + 1)\"), (\"4\")) + (\"7\")",
			CreateBinary(
				new List(new Body(method),
					GetListExpressions([
						"(\"Hello, World\", \"Yoyo (it is my secret + 1)\"), (\"4\")"
					])), BinaryOperator.Plus,
				new List(new Body(method), [new Text(method, "7")])));

	[Test]
	public void ContainsMethodCallOnNumbersList()
	{
		var ifExpression = ParseExpression(
			"if 2 is in (1, 2, 3)",
			"\tconstant abc = \"abc\"",
			"\tlogger.Log(abc)") as If;
		var numbers = (ifExpression?.Condition as MethodCall)?.Instance as List;
		Assert.That(numbers?.ToString(), Is.EqualTo("(1, 2, 3)"));
	}

	[Test]
	public void MethodsAndMembersOfListShouldHaveImplementationTypeAsParent()
	{
		var numbers = type.GetListImplementationType(type.GetType(Base.Number));
		Assert.That(numbers.Members[1].ToString(),
			Is.EqualTo("elements TestPackage.List(Number)"));
		Assert.That(numbers.Methods[1].Parent.ToString(),
			Is.EqualTo("TestPackage.List(Number)"));
	}

	[Test]
	public void MethodBodyShouldBeUpdatedWithImplementationType()
	{
		var texts = type.GetListImplementationType(type.GetType(Base.Text));
		var containsMethod = texts.Methods.FirstOrDefault(m =>
			m.Name == BinaryOperator.In && m.Parameters[0].Type.Name == Base.Text);
		Assert.That(containsMethod!.Type, Is.EqualTo(texts));
		Assert.That(containsMethod.Parameters[0].Type.Name, Is.EqualTo(Base.Text));
		var body = (Body)containsMethod.GetBodyAndParseIfNeeded();
		Assert.That(body.Method, Is.EqualTo(containsMethod));
		Assert.That(body.Method.Type, Is.EqualTo(texts));
		Assert.That(body.Method.Parameters[0].Type.Name, Is.EqualTo(Base.Text));
		Assert.That(body.Method, Is.EqualTo(containsMethod), texts.Methods.ToWordList());
	}

	[Test]
	public void CompareLists() =>
		Assert.That(
			EqualsExtensions.AreEqual(new List<Number> { new(type, 2), new(type, 3) },
				new List<Number> { new(type, 2), new(type, 3) }), Is.True);

	[Test]
	public void Index() =>
		Assert.That(ParseExpression("(1, 2, 3).Index(9) is -1").ReturnType,
			Is.EqualTo(type.GetType(Base.Boolean)));
}