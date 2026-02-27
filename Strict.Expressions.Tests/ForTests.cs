using static Strict.Expressions.For;

namespace Strict.Expressions.Tests;

public sealed class ForTests : TestExpressions
{
	[Test]
	public void MissingBody() =>
		Assert.That(() => ParseExpression("for Range(2, 5)"),
			Throws.InstanceOf<MissingInnerBody>());

	[Test]
	public void MissingExpression() =>
		Assert.That(() => ParseExpression("for"), Throws.InstanceOf<MissingInnerBody>());

	[Test]
	public void VariableOutOfScope() =>
		Assert.That(
			() => ParseExpression("for Range(2, 5)", "\tconstant num = 5", "for Range(0, 10)",
				"\tlogger.Log(num)"),
			Throws.InstanceOf<Body.IdentifierNotFound>().With.Message.StartWith("num"));

	[Test]
	public void Equals()
	{
		var first = ParseExpression("for Range(2, 5)", "\tlogger.Log(\"Hi\")");
		var second = ParseExpression("for Range(2, 5)", "\tlogger.Log(\"Hi\")");
		Assert.That(first, Is.InstanceOf<For>());
		Assert.That(first.Equals(second), Is.True);
	}

	[Test]
	public void MatchingHashCode()
	{
		var forExpression = (For)ParseExpression("for Range(2, 5)", "\tlogger.Log(index)");
		Assert.That(forExpression.IsConstant, Is.False);
		Assert.That(forExpression.GetHashCode(), Is.EqualTo(forExpression.Iterator.GetHashCode()));
	}

	[Test]
	public void IndexIsReserved() =>
		Assert.That(() => ParseExpression("for index in Range(0, 5)", "\tlogger.Log(index)"),
			Throws.InstanceOf<For.IndexIsReservedDoNotUseItExplicitly>());

	[Test]
	public void ForVariableMatchingMemberIsNotAddedAsVariable() =>
		Assert.That(() => ParseExpression("for five in (1, 2, 3)", "\tlogger.Log(five)"),
			Throws.InstanceOf<Body.IdentifierNotFound>());

	[TestCase("for gibberish", "\tlogger.Log(\"Hi\")")]
	[TestCase("for element in gibberish", "\tlogger.Log(element)")]
	public void UnidentifiedIterable(params string[] lines) =>
		Assert.That(() => ParseExpression(lines),
			Throws.InstanceOf<Body.IdentifierNotFound>());

	[Test]
	public void ImmutableVariableNotAllowedToBeAnIterator() =>
		Assert.That(
			() => ParseExpression("constant myIndex = 0", "for myIndex in Range(0, 10)",
				"\tlogger.Log(myIndex)"), Throws.InstanceOf<For.ImmutableIterator>());

	[Test]
	public void IteratorTypeDoesNotMatchWithIterable() =>
		Assert.That(
			() => ParseExpression("mutable element = 0", "for element in (\"1\", \"2\", \"3\")",
				"\tlogger.Log(element)"),
			Throws.InstanceOf<IteratorTypeDoesNotMatchWithIterable>().With.Message.Contains(
				"Iterator element type Text does not match with Number"));

	[Test]
	public void ForVariableUsesNonListIteratorValue()
	{
		var programType = new Type(type.Package,
				new TypeLines(nameof(ForVariableUsesNonListIteratorValue), "has logger",
					"LogCount(count Number) Number",
					"\tfor element in count",
					"\t\tlogger.Log(element)",
					"\t\telement",
					"\tcount")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var body = (Body)programType.Methods[0].GetBodyAndParseIfNeeded();
		var forExpression = (For)body.Expressions[0];
		Assert.That(((Body)forExpression.Body).FindVariable("element")?.Type.Name,
			Is.EqualTo(Base.Number));
	}

	[Test]
	public void ParseNestedForExpression() =>
		Assert.That(
			((For)ParseExpression("for Range(1, 3)", "\tfor Range(1, 3)",
				"\t\tlogger.Log(index + outer.index)")).ToString(),
			Is.EqualTo("for Range(1, 3)" + Environment.NewLine + "\tfor Range(1, 3)" +
				Environment.NewLine + "\t\tlogger.Log(index + outer.index)"));

	[Test]
	public void ParseForRangeExpression() =>
		Assert.That(((For)ParseExpression("for Range(2, 5)", "\tlogger.Log(index)")).ToString(),
			Is.EqualTo("for Range(2, 5)" + Environment.NewLine + "\tlogger.Log(index)"));

	[Test]
	public void ParseForDictionaryElementsExpression()
	{
		var number = type.GetType(Base.Number);
		var dictionary = type.GetType(Base.Dictionary).GetGenericImplementation(number, number);
		var runMethod = new Method(dictionary, 0, new MethodExpressionParser(), [
			"Run Number",
			"\tfor elements",
			"\t\t1"
		]);
		Assert.That(runMethod.GetBodyAndParseIfNeeded().ToString(),
			Is.EqualTo("for elements" + Environment.NewLine + "\t1"));
	}

	[Test]
	public void ParseForInExpression() =>
		Assert.That(
			((For)((Body)ParseExpression("mutable myIndex = 0", "for myIndex in Range(0, 5)",
				"\tlogger.Log(myIndex)")).Expressions[1]).ToString(),
			Is.EqualTo("for myIndex in Range(0, 5)" + Environment.NewLine + "\tlogger.Log(myIndex)"));

	[TestCase("for myIndex in Range(0, 5)", "\tlogger.Log(myIndex)",
		"for myIndex in Range(0, 5)\n\tlogger.Log(myIndex)")]
	[TestCase("for (1, 2, 3)", "\tlogger.Log(index)", "for (1, 2, 3)\n\tlogger.Log(index)")]
	[TestCase("for (1, 2, 3)", "\tlogger.Log(value)", "for (1, 2, 3)\n\tlogger.Log(value)")]
	[TestCase("for myIndex in Range(2, 5)", "\tlogger.Log(myIndex)", "\tfor Range(0, 10)",
		"\t\tlogger.Log(index)",
		"for myIndex in Range(2, 5)\n" + "\tlogger.Log(myIndex)\n" + "\tfor Range(0, 10)\n" +
		"\t\tlogger.Log(index)")]
	[TestCase("for firstIndex in Range(1, 10)", "\tfor secondIndex in Range(1, 10)",
		"\t\tlogger.Log(firstIndex)", "\t\tlogger.Log(secondIndex)",
		"for firstIndex in Range(1, 10)\n" + "\tfor secondIndex in Range(1, 10)\n" +
		"\t\tlogger.Log(firstIndex)\n" + "\t\tlogger.Log(secondIndex)")]
	public void ParseForExpressionWithCustomVariableName(params string[] lines) =>
		Assert.That(((For)ParseExpression(lines[..^1])).ToString(),
			Is.EqualTo(lines[^1].Replace("\n", Environment.NewLine)));

	[Test]
	public void NestedIfInForIsIndented() =>
		Assert.That(
			((For)ParseExpression("for Range(0, 2)", "\tif five is 5", "\t\tlogger.Log(\"Hey\")")).
			ToString(),
			Is.EqualTo("for Range(0, 2)" + Environment.NewLine + "\tif five is 5" +
				Environment.NewLine + "\t\tlogger.Log(\"Hey\")"));

	[Test]
	public void ValidIteratorReturnTypeWithValue() =>
		Assert.That(
			((VariableCall)((MethodCall)((For)ParseExpression("for (1, 2, 3)", "\tlogger.Log(value)")).
				Body).Arguments[0]).ReturnType.FullName, Is.EqualTo("TestPackage.Number"));

	[TestCase("constant elements = (1, 2, 3)", "for elements", "\tlogger.Log(index)",
		"for elements\n\tlogger.Log(index)")]
	[TestCase("constant elements = (1, 2, 3)", "for Range(0, elements.Length)", "\tlogger.Log(index)",
		"for Range(0, elements.Length)\n\tlogger.Log(index)")]
	[TestCase("mutable element = 0", "for element in (1, 2, 3)", "\tlogger.Log(element)",
		"for element in (1, 2, 3)\n\tlogger.Log(element)")]
	[TestCase("constant iterationCount = 10", "for iterationCount", "\tlogger.Log(index)",
		"for iterationCount\n\tlogger.Log(index)")]
	[TestCase("constant dummy = 0", "for 10", "\tlogger.Log(index)",
		"for 10\n\tlogger.Log(index)")]
	[TestCase("mutable element = \"1\"", "for element in (\"1\", \"2\", \"3\")",
		"\tlogger.Log(element)", "for element in (\"1\", \"2\", \"3\")\n\tlogger.Log(element)")]
	public void ParseForListExpressionWithIterableVariable(params string[] lines) =>
		Assert.That(((For)((Body)ParseExpression(lines[..^1])).Expressions[1]).ToString(),
			Is.EqualTo(lines[^1].Replace("\n", Environment.NewLine)));

	[Test]
	public void ValidIteratorReturnTypeForRange() =>
		Assert.That(
			((MethodCall)((For)ParseExpression("for Range(0, 10)", "\tlogger.Log(index)")).Body).
			Arguments[0].ReturnType.IsNumber);

	[Test]
	public void ValidIteratorReturnTypeTextForList() =>
		Assert.That(
			((VariableCall)((MethodCall)((For)((Body)ParseExpression("mutable element = \"1\"",
					"for element in (\"1\", \"2\", \"3\")", "\tlogger.Log(element)")).Expressions[1]).Body).
				Arguments[0]).Variable.Type.IsText);

	[Test]
	public void ValidLoopProgram()
	{
		using var programType = new Type(type.Package,
				new TypeLines(Base.App, "has number",
					"CountNumber Number",
					"\tfor Range(0, number)",
					"\t\t1")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var parsedExpression = (For)programType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(parsedExpression.ReturnType.Name, Is.EqualTo(Base.Range));
		Assert.That(parsedExpression.Iterator.ToString(), Is.EqualTo("Range(0, number)"));
	}

	[Test]
	public void ErrorExpressionIsNotAnIterator()
	{
		var programType = new Type(type.Package,
				new TypeLines(nameof(ErrorExpressionIsNotAnIterator), "has number", "LogError Number",
					"\tconstant error = Error(\"Process Failed\")", "\tfor error", "\t\tvalue")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(() => programType.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<For.ExpressionTypeIsNotAnIterator>());
	}

	[TestCase("error.Stacktraces", nameof(IterateErrorTypeMembers) + "StackTrace")]
	[TestCase("error.Text", nameof(IterateErrorTypeMembers) + "Text")]
	public void IterateErrorTypeMembers(string forExpressionText, string testName)
	{
		var programType = new Type(type.Package,
			new TypeLines(testName, "has number", "LogError Number",
				"\tconstant error = Error(\"Process Failed\")", $"\tfor {forExpressionText}",
				"\t\tvalue")).ParseMembersAndMethods(new MethodExpressionParser());
		var parsedExpression = (Body)programType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(parsedExpression.Expressions[1], Is.TypeOf(typeof(For)));
		Assert.That(((For)parsedExpression.Expressions[1]).Iterator.ToString(),
			Is.EqualTo(forExpressionText));
	}

	[Test]
	public void IterateNameType()
	{
		var programType = new Type(type.Package,
				new TypeLines(nameof(IterateNameType), "has number", "LogError Number",
					"\tconstant name = Name(\"Strict\")", "\tfor name", "\t\tvalue")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var parsedExpression = (Body)programType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(parsedExpression.Expressions[1], Is.TypeOf(typeof(For)));
		Assert.That(((For)parsedExpression.Expressions[1]).Iterator.ToString(), Is.EqualTo("name"));
	}

	[Test]
	public void MissingBodyInNestedFor() =>
		Assert.That(() => ParseExpression(
				"for Range(2, 5)",
				"for index in Range(1, 10)"),
			Throws.InstanceOf<For.MissingInnerBody>());

	[TestCase(
		"WithParameter", "for element in (1, 2, 3, 4)",
		"has logger",
		"LogError Number",
		"\tfor element in (1, 2, 3, 4)",
		"\t\tlogger.Log(element)")]
	[TestCase(
		"WithList", "for element in elements",
		"has logger",
		"LogError(elements Numbers) Number",
		"\tfor element in elements",
		"\t\tlogger.Log(element)")]
	[TestCase(
		"WithListTexts", "for element in texts",
		"has logger",
		"LogError(texts) Number",
		"\tfor element in texts",
		"\t\tlogger.Log(element)")]
	public void AllowCustomVariablesInFor(string testName, string expected, params string[] code)
	{
		var programType =
			new Type(type.Package, new TypeLines(nameof(AllowCustomVariablesInFor) + testName, code)).
				ParseMembersAndMethods(new MethodExpressionParser());
		var parsedExpression = (For)programType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(parsedExpression.ToString(), Does.StartWith(expected));
	}

	[TestCase("WithNumbers", "for row, column in listOfNumbers",
		"has logger",
		"LogAllNumbers(listOfNumbers List(Numbers))",
		"\tfor row, column in listOfNumbers",
		"\t\tlogger.Log(column)")]
	[TestCase(
		"WithTexts", "for row, column in texts",
		"has logger",
		"LogTexts(texts)",
		"\tfor row, column in texts",
		"\t\tlogger.Log(column)")]
	public void ParseForExpressionWithMultipleVariables(string testName, string expected, params string[] code)
	{
		var programType =
			new Type(type.Package,
					new TypeLines(nameof(ParseForExpressionWithMultipleVariables) + testName, code)).
				ParseMembersAndMethods(new MethodExpressionParser());
		var parsedExpression = (For)programType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(parsedExpression.ToString(), Does.StartWith(expected));
	}

	[Test]
	public void ForBodyWithInlineConditionalThenConcatenation()
	{
		var programType = new Type(type.Package,
				new TypeLines(nameof(ForBodyWithInlineConditionalThenConcatenation), "has number",
					"GetElementsText(elements Numbers) Text",
					"\tfor elements",
					"\t\t(index is 0 then \"\" else \", \") + value")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var forExpression = (For)programType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(forExpression.ToString(), Does.StartWith("for elements"));
	}

	[Test]
	public void RemoveParenthesesWithElseIfChain()
	{
		var programType = new Type(type.Package,
				new TypeLines(nameof(RemoveParenthesesWithElseIfChain), "has text",
					"Remove Text",
					"\tmutable parentheses = 0",
					"\tfor text",
					"\t\tif value is \"(\"",
					"\t\t\tparentheses = parentheses + 1",
					"\t\telse if value is \")\"",
					"\t\t\tparentheses = parentheses - 1",
					"\t\telse if parentheses is 0",
					"\t\t\tvalue")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var body = (Body)programType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.Expressions[1], Is.TypeOf<For>());
	}

	[Test]
	public void ForBodyMultiLinePiping()
	{
		var programType = new Type(type.Package,
				new TypeLines(nameof(ForBodyMultiLinePiping), "has numbers",
					"GetNumbersText Numbers",
					"\tfor numbers",
					"\t\tto Text",
					"\t\tLength")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var forExpression = (For)programType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(forExpression.Body.ToString(), Does.Contain("to Text"));
	}
}