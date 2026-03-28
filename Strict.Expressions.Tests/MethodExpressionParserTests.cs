using Strict.Language.Tests;

namespace Strict.Expressions.Tests;

public sealed class MethodExpressionParserTests : TestExpressions
{
	[Test]
	public void CannotParseEmptyInputException() =>
		Assert.That(() => new MethodExpressionParser().ParseExpression(new Body(method), ""),
			Throws.InstanceOf<CannotParseEmptyInput>());

	[Test]
	public void ParseSingleLine()
	{
		var body = (Body)new Method(type, 0, this,
				[MethodTests.Run, MethodTests.ConstantNumber, "\tnumber is Number"]).
			GetBodyAndParseIfNeeded();
		var declaration = (Declaration)body.Expressions[0];
		Assert.That(declaration.ReturnType, Is.EqualTo(type.FindType(Type.Number)));
		Assert.That(declaration.ToString(), Is.EqualTo(MethodTests.ConstantNumber[1..]));
	}

	[Test]
	public void ParseMultipleLines()
	{
		var body = (Body)new Method(type, 0, this, [
			MethodTests.Run, MethodTests.ConstantNumber, MethodTests.ConstantOther
		]).GetBodyAndParseIfNeeded();
		Assert.That(body.Expressions, Has.Count.EqualTo(2));
		Assert.That(body.Expressions[0].ToString(), Is.EqualTo(MethodTests.ConstantNumber[1..]));
		Assert.That(body.Expressions[1].ToString(), Is.EqualTo(MethodTests.ConstantOther[1..]));
	}

	[Test]
	public void ParseNestedLines()
	{
		var body = (Body)new Method(type, 0, this, MethodTests.NestedMethodLines).
			GetBodyAndParseIfNeeded();
		Assert.That(body.Expressions, Has.Count.EqualTo(3));
		Assert.That(body.Expressions[0].ToString(), Is.EqualTo(MethodTests.ConstantNumber[1..]));
		Assert.That(body.Expressions[1].ToString(),
			Is.EqualTo(MethodTests.NestedMethodLines[2][1..] + Environment.NewLine +
				MethodTests.NestedMethodLines[3][1..]));
		Assert.That(body.Expressions[2].ToString(), Is.EqualTo("false"));
	}

	[TestCase("\tError(errorMessage)")]
	[TestCase("\tError(\"error occurred\")")]
	[TestCase("\tError(\"error occurred: \" + errorMessage)")]
	[TestCase("\tError(\"error occurred: \" + errorMessage + \"at line\")")]
	[TestCase("\tError(\"error occurred: \" + errorMessage + \"at line\" + \"5\")")]
	public void ParseErrorExpression(string errorExpression)
	{
		var body = (Body)new Method(type, 0, this, [
			MethodTests.Run, MethodTests.ConstantErrorMessage, errorExpression
		]).GetBodyAndParseIfNeeded();
		Assert.That(body.ReturnType, Is.EqualTo(type.FindType(Type.None)));
		Assert.That(body.Expressions, Has.Count.EqualTo(2));
		Assert.That(body.Expressions[0].ReturnType.Name, Is.EqualTo(Type.Text));
		Assert.That(body.Expressions[1], Is.TypeOf<MethodCall>());
	}

	[Test]
	public void ParseInlineTextLiteralMethodCallAssertion()
	{
		using var parsingType = new Type(TestPackage.Instance, new TypeLines(
			nameof(ParseInlineTextLiteralMethodCallAssertion),
			"has number",
			"LastIndexOf(text Text) Number",
			"\t\"hello\".LastIndexOf(\"l\") is 3",
			"\t-1")).ParseMembersAndMethods(this);
		Assert.That(() => parsingType.Methods.Single().GetBodyAndParseIfNeeded(), Throws.Nothing);
	}

	[Test]
	public void IsVariableMutatedInNestedBody()
	{
		var body = (Body)new Method(type, 0, this, [
			MethodTests.Run,
			"\tmutable result = 0",
			"\tfor Range(0, 10)",
			"\t\tresult = result + 1",
			"\tresult"
		]).GetBodyAndParseIfNeeded();
		Assert.That(body.Method.Parser.IsVariableMutated(body, "result"), Is.True);
	}

	[Test]
	public void IsVariableMutatedInIfThen()
	{
		var body = (Body)new Method(type, 0, this, [
			MethodTests.Run,
			"\tconstant number = 5",
			"\tmutable result = 0",
			"\tif number is 5",
			"\t\tresult = 1",
			"\tresult"
		]).GetBodyAndParseIfNeeded();
		Assert.That(body.Method.Parser.IsVariableMutated(body, "result"), Is.True);
	}

	[Test]
	public void IsVariableMutatedInIfElse()
	{
		var body = (Body)new Method(type, 0, this, [
			MethodTests.Run,
			"\tconstant number = 5",
			"\tmutable result = 0",
			"\tif number is 5",
			"\t\treturn 1",
			"\telse",
			"\t\tresult = 2",
			"\tresult"
		]).GetBodyAndParseIfNeeded();
		Assert.That(body.Method.Parser.IsVariableMutated(body, "result"), Is.True);
	}

	[Test]
	public void IsVariableMutatedInNestedIfBody()
	{
		var body = (Body)new Method(type, 0, this, [
			MethodTests.Run,
			"\tconstant number = 5",
			"\tmutable result = 0",
			"\tif number is 5",
			"\t\tif true",
			"\t\t\tresult = 1",
			"\tresult"
		]).GetBodyAndParseIfNeeded();
		Assert.That(body.Method.Parser.IsVariableMutated(body, "result"), Is.True);
	}

	[Test]
	public void IsVariableMutatedInNestedElseBody()
	{
		var body = (Body)new Method(type, 0, this, [
			MethodTests.Run,
			"\tconstant number = 5",
			"\tmutable result = 0",
			"\tif number is 5",
			"\t\treturn 1",
			"\telse",
			"\t\tif true",
			"\t\t\tresult = 2",
			"\tresult"
		]).GetBodyAndParseIfNeeded();
		Assert.That(body.Method.Parser.IsVariableMutated(body, "result"), Is.True);
	}

	[Test]
	public void IsVariableMutatedInListCall()
	{
		var body = (Body)new Method(type, 0, this, [
			MethodTests.Run,
			"\tmutable result = (1, 2)",
			"\tresult(0) = 0",
			"\tresult"
		]).GetBodyAndParseIfNeeded();
		Assert.That(body.Method.Parser.IsVariableMutated(body, "result"), Is.True);
	}

	[Test]
	public async Task GenericListPlusMethodShouldParseWithoutGenericLookupError()
	{
		var listPlus = TestPackage.Instance.GetType(Type.List).Methods.Single(m =>
			m.Name == BinaryOperator.Plus && m.Parameters[0].Type.IsList);
		Assert.That(() => listPlus.GetBodyAndParseIfNeeded(), Throws.Nothing);
	}

	[Test]
	public void ParseListLiteralContainingTextWithBrackets()
	{
		var body = (Body)new Method(type, 0, this, [
			MethodTests.Run,
			"\tconstant expected = (\"3\", \"4\", \"(1, 2)\")",
			"\texpected"
		]).GetBodyAndParseIfNeeded();
		var declaration = (Declaration)body.Expressions[0];
		Assert.That(declaration.Value, Is.InstanceOf<List>());
		Assert.That(declaration.Value.ToString(),
			Is.EqualTo("(\"3\", \"4\", \"(1, 2)\")"));
	}

	[Test]
	public void ParseIsComparisonWithRightSideListContainingTextWithBrackets()
	{
		var body = (Body)new Method(type, 0, this, [
			MethodTests.Run,
			"\tconstant oneTwo = (1, 2) to Text",
			"\t(\"3\", \"4\") + oneTwo is (\"3\", \"4\", \"(1, 2)\")",
			"\ttrue"
		]).GetBodyAndParseIfNeeded();
		Assert.That(body.Expressions, Has.Count.EqualTo(3));
	}

	[Test]
	public async Task ParseListReverseMethod()
	{
		var listReverse = TestPackage.Instance.GetType(Type.List).Methods.Single(method =>
			method.Name == "Reverse" && method.Parameters.Count == 0);
		Assert.That(() => listReverse.GetBodyAndParseIfNeeded(), Throws.Nothing);
	}

	[Test]
	public async Task ParseHashCodeFromWithGenericTypeDispatch()
	{
		var basePackage = await new Repositories(new MethodExpressionParser()).LoadStrictPackage();
		var hashCodeType = basePackage.FindType("HashCode")!;
		var fromMethod = hashCodeType.Methods.Single(m => m.Name == Method.From);
		Assert.That(() => fromMethod.GetBodyAndParseIfNeeded(), Throws.Nothing,
			"HashCode.from selector-if on generic type should parse");
	}

	[Test]
	public void ParseRangeForIteratorMethod()
	{
		var forMethod = TestPackage.Instance.GetType(Type.Range).Methods.Single(m => m.Name == "for");
		Assert.That(() => forMethod.GetBodyAndParseIfNeeded(), Throws.Nothing,
			"Range.for Iterator(Number) method body should parse");
	}

	[Test]
	public async Task ParseEnumCompareToWithNestedConditional()
	{
		var basePackage = await new Repositories(new MethodExpressionParser()).LoadStrictPackage();
		var enumType = basePackage.FindType("Enum")!;
		var compareToMethod = enumType.Methods.Single(m => m.Name == "CompareTo");
		Assert.That(() => compareToMethod.GetBodyAndParseIfNeeded(), Throws.Nothing,
			"Enum.CompareTo nested conditional should parse");
	}

	[Test]
	public async Task ParseDictionaryInMethod()
	{
		var basePackage = await new Repositories(new MethodExpressionParser()).LoadStrictPackage();
		var dictionaryType = basePackage.FindType("Dictionary")!;
		var inMethod = dictionaryType.Methods.Single(m => m.Name == "in");
		Assert.That(() => inMethod.GetBodyAndParseIfNeeded(), Throws.Nothing,
			"Dictionary.in method with value.Key is key should parse");
	}

	[Test]
	public async Task ParseDictionaryGetMethod()
	{
		var basePackage = await new Repositories(new MethodExpressionParser()).LoadStrictPackage();
		var dictionaryType = basePackage.FindType("Dictionary")!;
		var getMethod = dictionaryType.Methods.Single(m => m.Name == "get");
		Assert.That(() => getMethod.GetBodyAndParseIfNeeded(), Throws.Nothing,
			"Dictionary.get method with return value.Value should parse");
	}

	[Test]
	public async Task ParseDictionaryAddMethod()
	{
		var basePackage = await new Repositories(new MethodExpressionParser()).LoadStrictPackage();
		var dictionaryType = basePackage.FindType("Dictionary")!;
		var addMethod = dictionaryType.Methods.Single(m => m.Name == "Add");
		Assert.That(() => addMethod.GetBodyAndParseIfNeeded(), Throws.Nothing,
			"Dictionary.Add method with keysAndValues.Add((key, mappedValue)) should parse");
	}

	[Test]
	public async Task ParseStacktraceToMethod()
	{
		var basePackage = await new Repositories(new MethodExpressionParser()).LoadStrictPackage();
		var stacktraceType = basePackage.FindType("Stacktrace")!;
		var toMethod = stacktraceType.Methods.Single(m => m.Name == BinaryOperator.To);
		Assert.That(() => toMethod.GetBodyAndParseIfNeeded(), Throws.Nothing,
			"Stacktrace.to method with escaped tab and backslash text should parse");
	}

	[Test]
	public async Task ParseAllStrictBasePackageCode()
	{
		var basePackage = await new Repositories(new MethodExpressionParser()).LoadStrictPackage();
		foreach (var baseType in new List<Type>(basePackage.Types.Values))
		foreach (var baseMethod in baseType.Methods)
			if (!baseMethod.IsTrait)
				Assert.That(() => baseMethod.GetBodyAndParseIfNeeded(), Throws.Nothing,
					$"Failed to parse method {baseMethod.Name} in type {baseType.Name}");
	}
}