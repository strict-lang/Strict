namespace Strict.Expressions.Tests;

public sealed class ForTests : TestExpressions
{
	[Test]
	public void MissingBody() =>
		Assert.That(() => ParseExpression("for Range(2, 5)"),
			Throws.InstanceOf<For.MissingInnerBody>());

	[Test]
	public void MissingExpression() =>
		Assert.That(() => ParseExpression("for"), Throws.InstanceOf<For.MissingExpression>());

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
		Assert.That(forExpression.GetHashCode(), Is.EqualTo(forExpression.Value.GetHashCode()));
	}

	[Test]
	public void IndexIsReserved() =>
		Assert.That(() => ParseExpression("for index in Range(0, 5)", "\tlogger.Log(index)"),
			Throws.InstanceOf<For.IndexIsReservedDoNotUseItExplicitly>());

	[TestCase("for gibberish", "\tlogger.Log(\"Hi\")")]
	[TestCase("for element in gibberish", "\tlogger.Log(element)")]
	public void UnidentifiedIterable(params string[] lines) =>
		Assert.That(() => ParseExpression(lines),
			Throws.InstanceOf<Body.IdentifierNotFound>());

	[Test]
	public void DuplicateImplicitIndexInNestedFor() =>
		Assert.That(
			() => ParseExpression("for Range(2, 5)", "\tfor Range(0, 10)", "\t\tlogger.Log(index)"),
			Throws.InstanceOf<For.DuplicateImplicitIndex>());

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
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>());

	[Test]
	public void ParseForRangeExpression() =>
		Assert.That(((For)ParseExpression("for Range(2, 5)", "\tlogger.Log(index)")).ToString(),
			Is.EqualTo("for Range(2, 5)\n\tlogger.Log(index)"));

	[Test]
	public void ParseForInExpression() =>
		Assert.That(
			((For)((Body)ParseExpression("mutable myIndex = 0", "for myIndex in Range(0, 5)",
				"\tlogger.Log(myIndex)")).Expressions[1]).ToString(),
			Is.EqualTo("for myIndex in Range(0, 5)\n\tlogger.Log(myIndex)"));

	[TestCase("for myIndex in Range(0, 5)", "\tlogger.Log(myIndex)",
		"for myIndex in Range(0, 5)\n\tlogger.Log(myIndex)")]
	[TestCase("for (1, 2, 3)", "\tlogger.Log(index)", "for (1, 2, 3)\n\tlogger.Log(index)")]
	[TestCase("for (1, 2, 3)", "\tlogger.Log(value)", "for (1, 2, 3)\n\tlogger.Log(value)")]
	[TestCase("for myIndex in Range(2, 5)", "\tlogger.Log(myIndex)", "\tfor Range(0, 10)",
		"\t\tlogger.Log(index)",
		"for myIndex in Range(2, 5)\n\tlogger.Log(myIndex)\r\nfor Range(0, 10)\n\tlogger.Log(index)")]
	[TestCase("for firstIndex in Range(1, 10)", "for secondIndex in Range(1, 10)",
		"\tlogger.Log(firstIndex)", "\tlogger.Log(secondIndex)",
		"for firstIndex in Range(1, 10)\n\tfor secondIndex in Range(1, 10)\n\tlogger.Log(firstIndex)\r\nlogger.Log(secondIndex)")]
	public void ParseForExpressionWithCustomVariableName(params string[] lines) =>
		Assert.That(((For)ParseExpression(lines[..^1])).ToString(), Is.EqualTo(lines[^1]));

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
			Is.EqualTo(lines[^1]));

	[Test]
	public void ValidIteratorReturnTypeForRange() =>
		Assert.That(
			((MethodCall)((For)ParseExpression("for Range(0, 10)", "\tlogger.Log(index)")).Body).
			Arguments[0].ReturnType.Name == Base.Number);

	[Test]
	public void ValidIteratorReturnTypeTextForList() =>
		Assert.That(
			((VariableCall)((MethodCall)((For)((Body)ParseExpression("mutable element = \"1\"",
					"for element in (\"1\", \"2\", \"3\")", "\tlogger.Log(element)")).Expressions[1]).Body).
				Arguments[0]).Variable.Type.Name == Base.Text);

	[Test]
	public void ValidLoopProgram()
	{
		using var programType = new Type(type.Package,
				new TypeLines(Base.App, "has number",
					"CountNumber Number",
					"\tmutable result = 1",
					"\tfor Range(0, number)",
					"\t\tresult = result + 1",
					"\tresult")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var parsedExpression = (Body)programType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(parsedExpression.ReturnType.Name, Is.EqualTo(Base.Number));
		Assert.That(parsedExpression.Expressions[1], Is.TypeOf(typeof(For)));
		Assert.That(((For)parsedExpression.Expressions[1]).Value.ToString(),
			Is.EqualTo("Range(0, number)"));
	}

	[Test]
	public void ErrorExpressionIsNotAnIterator()
	{
		var programType = new Type(type.Package,
				new TypeLines(nameof(ErrorExpressionIsNotAnIterator), "has number", "LogError Number", "\tconstant error = Error(\"Process Failed\")",
					"\tfor error", "\t\tvalue")).
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
		Assert.That(((For)parsedExpression.Expressions[1]).Value.ToString(),
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
		Assert.That(((For)parsedExpression.Expressions[1]).Value.ToString(), Is.EqualTo("name"));
	}

	[Test]
	public void MissingBodyInNestedFor() =>
		Assert.That(() => ParseExpression(
				"for Range(2, 5)",
				"for index in Range(1, 10)"),
			Throws.InstanceOf<For.MissingInnerBody>());

	[TestCase(
		"WithParameter", "element in (1, 2, 3, 4)",
		"has logger",
		"LogError Number",
		"\tfor element in (1, 2, 3, 4)",
		"\t\tlogger.Log(element)")]
	[TestCase(
		"WithList", "element in elements",
		"has logger",
		"LogError(elements Numbers) Number",
		"\tfor element in elements",
		"\t\tlogger.Log(element)")]
	[TestCase(
		"WithListTexts", "element in texts",
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
		Assert.That(parsedExpression.Value.ToString(), Is.EqualTo(expected));
	}

	[TestCase("WithNumbers",
		"has logger",
		"LogAllNumbers(listOfNumbers List(Numbers))",
		"\tfor row, column in listOfNumbers",
		"\t\tlogger.Log(column)")]
	[TestCase(
		"WithTexts",
		"has logger",
		"LogTexts(texts)",
		"\tfor row, column in texts",
		"\t\tlogger.Log(column)")]
	public void ParseForExpressionWithMultipleVariables(string testName, params string[] code)
	{
		var programType =
			new Type(type.Package,
					new TypeLines(nameof(ParseForExpressionWithMultipleVariables) + testName, code)).
				ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(programType.Methods[0].GetBodyAndParseIfNeeded(), Is.InstanceOf<For>());
	}
}