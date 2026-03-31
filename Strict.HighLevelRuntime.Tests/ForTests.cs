using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime.Tests;

public sealed class ForTests
{
	[SetUp]
	public void CreateExecutor() =>
		interpreter = new Interpreter(TestPackage.Instance, TestBehavior.Disabled);

	private Interpreter interpreter = null!;

	private static Type CreateType(string name, params string[] lines) =>
		new Type(TestPackage.Instance, new TypeLines(name, lines)).ParseMembersAndMethods(
			new MethodExpressionParser());

	[Test]
	public async Task SizeIterationOrderMatchesTwoDimensionalColorImageIndexing()
	{
		var parser = new MethodExpressionParser();
		var repositories = new Repositories(parser);
		using var strictPackage = await repositories.LoadStrictPackage();
		using var mathPackage = await repositories.LoadStrictPackage("Strict/Math");
		using var imageProcessingPackage = await repositories.LoadStrictPackage("Strict/ImageProcessing");
		using var testType = new Type(imageProcessingPackage,
			new TypeLines("ColorImageIndexing",
				"has number",
				"CenterIsExpectedColor Boolean",
				"\tconstant width = 2",
				"\tconstant height = 2",
				"\tconstant colors = (Color(0, 0, 0),",
				"\tColor(0.25, 0.25, 0.25),",
				"\tColor(0.5, 0.5, 0.5),",
				"\tColor(0.75, 0.75, 0.75))",
				"\tconstant image = ColorImage(Size(width, height), colors)",
				"\timage.Colors(width / 2, height / 2) is Color(0.75, 0.75, 0.75)",
				"Indices Numbers",
				"\tfor Size(2, 2)",
				"\t\tvalue.X + value.Y * 10")).ParseMembersAndMethods(parser);
		var packageInterpreter = new Interpreter(imageProcessingPackage, TestBehavior.Disabled);
		Assert.That(packageInterpreter.Execute(
			testType.Methods.Single(method => method.Name == "CenterIsExpectedColor"),
			packageInterpreter.noneInstance, []).Boolean, Is.True);
		Assert.That(packageInterpreter.Execute(testType.Methods.Single(method => method.Name == "Indices"),
			packageInterpreter.noneInstance, []).ToExpressionCodeString(), Is.EqualTo("(0, 1, 10, 11)"));
	}

	[Test]
	public void CustomVariableInForLoopIsUsed()
	{
		using var t = CreateType(nameof(CustomVariableInForLoopIsUsed), "has number", "Sum Number",
			"\tfor item in (1, 2, 3)", "\t\titem");
		var result = interpreter.Execute(t.Methods.Single(m => m.Name == "Sum"),
			interpreter.noneInstance, []);
		Assert.That(result.Number, Is.EqualTo(6));
	}

	[TestCase("add", 6)]
	[TestCase("subtract", -6)]
	public void SelectorIfInForUsesOperation(string operation, double expected)
	{
		using var t = CreateType(nameof(SelectorIfInForUsesOperation), "has operation Text",
			"Run Number", "\tfor (1, 2, 3)", "\t\tif operation is", "\t\t\t\"add\" then value",
			"\t\t\t\"subtract\" then 0 - value");
		var instance = new ValueInstance(t, [new ValueInstance(operation)]);
		var result = interpreter.Execute(t.Methods.Single(m => m.Name == "Run"), instance, []);
		Assert.That(result.Number, Is.EqualTo(expected));
	}

	[Test]
	public void TextReturnTypeConsolidatesNumbers()
	{
		using var t = CreateType(nameof(TextReturnTypeConsolidatesNumbers), "has number", "Join Text",
			"\tfor (\"a\", \"abc\")", "\t\tvalue.Length");
		var result = interpreter.Execute(t.Methods.Single(m => m.Name == "Join"),
			interpreter.noneInstance, []);
		Assert.That(result.Text, Is.EqualTo("13"));
	}

	[Test]
	public void TextReturnTypeConsolidatesTexts()
	{
		using var t = CreateType(nameof(TextReturnTypeConsolidatesTexts), "has number", "Join Text",
			"\tfor (\"hello \", \"world\")", "\t\tvalue");
		var result = interpreter.Execute(t.Methods.Single(m => m.Name == "Join"),
			interpreter.noneInstance, []);
		Assert.That(result.Text, Is.EqualTo("hello world"));
	}

	[Test]
	public void TextReturnTypeConsolidatesCharacters()
	{
		using var t = CreateType(nameof(TextReturnTypeConsolidatesCharacters), "has number",
			"Join(character) Text", "\tfor (character, character)", "\t\tvalue");
		var result = interpreter.Execute(t.Methods.Single(m => m.Name == "Join"),
			interpreter.noneInstance, [new ValueInstance(t.GetType(Type.Character), 'b')]);
		Assert.That(result.Text, Is.EqualTo("bb"));
	}

	[Test]
	public void TextReturnTypeThrowsWhenUnsupportedValueIsUsed()
	{
		using var t = CreateType(nameof(TextReturnTypeThrowsWhenUnsupportedValueIsUsed), "has number",
			"Join Text", "\tfor (1, 2)", "\t\tError(\"boom\")");
		Assert.That(
			() => interpreter.Execute(t.Methods.Single(m => m.Name == "Join"), interpreter.noneInstance,
				[]),
			Throws.InstanceOf<InterpreterExecutionFailed>().With.Message.
				Contains("For text return type cannot consolidate value"));
	}

	[Test]
	public void ForLoopUsesNumberIteratorLength()
	{
		using var t = CreateType(nameof(ForLoopUsesNumberIteratorLength), "has number", "Run Number",
			"\tfor 3", "\t\t1");
		var result = interpreter.Execute(t.Methods.Single(m => m.Name == "Run"),
			interpreter.noneInstance, []);
		Assert.That(result.Number, Is.EqualTo(3));
	}

	[Test]
	public void ForLoopUsesFloatingNumberIteratorLength()
	{
		using var t = CreateType(nameof(ForLoopUsesFloatingNumberIteratorLength), "has number",
			"Run Number", "\tfor 2.5", "\t\t1");
		var result = interpreter.Execute(t.Methods.Single(m => m.Name == "Run"),
			interpreter.noneInstance, []);
		Assert.That(result.Number, Is.EqualTo(2));
	}

	[Test]
	public void ForLoopThrowsWhenIteratorLengthIsUnsupported()
	{
		const string TypeName = nameof(ForLoopThrowsWhenIteratorLengthIsUnsupported);
		using var t = CreateType(TypeName, "has numbers", $"Run(container {TypeName}) Number",
			"\tfor container", "\t\t1");
		var container = new ValueInstance(t,
			[new ValueInstance(t.Members[0].Type, Array.Empty<ValueInstance>())]);
		Assert.That(
			() => interpreter.Execute(t.Methods.Single(m => m.Name == "Run"), interpreter.noneInstance,
				[container]), Throws.InstanceOf<ValueInstance.IteratorNotSupported>());
	}

	[Test]
	public void GetElementsTextWithCompactConditionalThen()
	{
		using var t = CreateType(nameof(GetElementsTextWithCompactConditionalThen), "has number",
			"GetElementsText(elements Numbers) Text", "\tfor elements",
			"\t\t(index is 0 then \"\" else \", \") + value");
		ValueInstance[] nums = [new(interpreter.numberType, 1), new(interpreter.numberType, 3)];
		var listType = interpreter.listType.GetGenericImplementation(interpreter.numberType);
		var result = interpreter.Execute(t.Methods.Single(m => m.Name == "GetElementsText"),
			interpreter.noneInstance, [new ValueInstance(listType, nums)]);
		Assert.That(result.Text, Is.EqualTo("1, 3"));
	}

	[Test]
	public void DirectOuterIndexerUsesImmediateParentValue()
	{
		using var t = CreateType(nameof(DirectOuterIndexerUsesImmediateParentValue), "has number",
			"Get(number, length Number) Text", "\tfor Range(number, number + length)",
			"\t\touter(value)");
		var instance = new ValueInstance(t, [new ValueInstance("hello")]);
		var result = interpreter.Execute(t.Methods.Single(m => m.Name == "Get"), instance,
		[
			new ValueInstance(interpreter.numberType, 1), new ValueInstance(interpreter.numberType, 3)
		]);
		Assert.That(result.Text, Is.EqualTo("ell"));
	}

	[Test]
	public void ForLoopWithAscendingRangeAndEarlyReturn()
	{
		using var t = CreateType(nameof(ForLoopWithAscendingRangeAndEarlyReturn), "has number",
			"FindFirst Number", "\tfor Range(1, 5)", "\t\tif value is 3", "\t\t\treturn value",
			"\t\t0");
		var result = interpreter.Execute(t.Methods.Single(m => m.Name == "FindFirst"),
			interpreter.noneInstance, []);
		Assert.That(result.Number, Is.EqualTo(3));
	}

	[Test]
	public void ForLoopWithDescendingRangeAndEarlyReturn()
	{
		using var t = CreateType(nameof(ForLoopWithDescendingRangeAndEarlyReturn), "has number",
			"FindFirst Number", "\tfor Range(5, 1)", "\t\tif value is 3", "\t\t\treturn value",
			"\t\t0");
		var result = interpreter.Execute(t.Methods.Single(m => m.Name == "FindFirst"),
			interpreter.noneInstance, []);
		Assert.That(result.Number, Is.EqualTo(3));
	}

	[Test]
	public void TextReturnTypeWithNoResultsReturnsEmpty()
	{
		using var t = CreateType(nameof(TextReturnTypeWithNoResultsReturnsEmpty), "has number",
			"GetText Text", "\tmutable sum = 0", "\tfor (1, 2, 3)", "\t\tif value > 0",
			"\t\t\tsum = sum + value");
		var result = interpreter.Execute(t.Methods.Single(m => m.Name == "GetText"),
			interpreter.noneInstance, []);
		Assert.That(result.Text, Is.EqualTo(""));
	}

	[Test]
	public void TextReturnTypeConsolidatesListValues()
	{
		using var t = CreateType(nameof(TextReturnTypeConsolidatesListValues), "has number",
			"Merge Text", "\tfor (\"a\", \"b\")", "\t\t(value, value)");
		var result = interpreter.Execute(t.Methods.Single(m => m.Name == "Merge"),
			interpreter.noneInstance, []);
		Assert.That(result.Text, Is.EqualTo("((a, a)), ((b, b))"));
	}

	[Test]
	public void RemoveParenthesesWithElseIfChain()
	{
		using var t = CreateType(nameof(RemoveParenthesesWithElseIfChain),
			// @formatter:off
			"has text",
			"Remove Text",
			"\tmutable parentheses = 0",
			"\tfor text",
			"\t\tif value is \"(\"",
			"\t\t\tparentheses.Increment",
			"\t\tif value is \")\"",
			"\t\t\tparentheses.Decrement",
			"\t\telse if parentheses is 0",
			"\t\t\tvalue");
		var result = interpreter.Execute(t.Methods.Single(m => m.Name == "Remove"),
			new ValueInstance(t, [ new ValueInstance("example(unwanted)example") ]), []);
		Assert.That(result.Text, Is.EqualTo("exampleexample"));
	}

	[Test]
	public void CountTextsWithLastIndexOfInForLoop()
	{
		using var t = CreateType(nameof(CountTextsWithLastIndexOfInForLoop),
			"has texts",
			"CountStartingWithHas Number",
			"\tfor texts",
			"\t\tif LastIndexOf(\"has \") is 0",
			"\t\t\t1");
		var textsType = t.Members[0].Type;
		var lines = new ValueInstance(textsType,
		[
			new ValueInstance("has name Text"),
			new ValueInstance("has count Number"),
			new ValueInstance("to Text")
		]);
		var instance = new ValueInstance(t, [lines]);
		var result = interpreter.Execute(t.Methods.Single(m => m.Name == "CountStartingWithHas"),
			instance, []);
		Assert.That(result.Number, Is.EqualTo(2));
	}

	[Test]
	public void CountTextsWithStartsWithInForLoop()
	{
		using var t = CreateType(nameof(CountTextsWithStartsWithInForLoop),
			"has texts",
			"CountStartingWithHas Number",
			"\tfor texts",
			"\t\tif StartsWith(\"has \")",
			"\t\t\t1");
		var textsType = t.Members[0].Type;
		var lines = new ValueInstance(textsType,
		[
			new ValueInstance("has name Text"),
			new ValueInstance("has count Number"),
			new ValueInstance("to Text")
		]);
		var instance = new ValueInstance(t, [lines]);
		var result = interpreter.Execute(t.Methods.Single(m => m.Name == "CountStartingWithHas"),
			instance, []);
		Assert.That(result.Number, Is.EqualTo(2));
	}

	[Test]
	public void LastIndexOfOnForLoopTextVariable()
	{
		using var t = CreateType(nameof(LastIndexOfOnForLoopTextVariable),
			"has texts",
			"Run Number",
			"\tfor texts",
			"\t\tLastIndexOf(\"h\")");
		var textsType = t.Members[0].Type;
		var instance = new ValueInstance(t,
			[new ValueInstance(textsType, [new ValueInstance("hello")])]);
		var result = interpreter.Execute(t.Methods.Single(m => m.Name == Method.Run),
			instance, []);
		Assert.That(result.Number, Is.EqualTo(0));
	}

	[Test]
	public void ForLoopMultiplication()
	{
		using var t = CreateType(nameof(LastIndexOfOnForLoopTextVariable),
			"has numbers",
			"Run Number",
			"\tfor numbers",
			"\t\t* value");
		var numberType = t.GetType(Type.Number);
		var result = interpreter.Execute(t.Methods.Single(m => m.Name == Method.Run),
			new ValueInstance(t, [new ValueInstance(t.Members[0].Type,
			[new ValueInstance(numberType, 2),
			new ValueInstance(numberType, 3),
			new ValueInstance(numberType, 4)])]), []);
		Assert.That(result.Number, Is.EqualTo(2 * 3 * 4));
	}

	[Test]
	public void StrictTypeParserCountsMembers()
	{
		using var typeParser = CreateType(nameof(StrictTypeParserCountsMembers),
			"has lines Texts",
			"MemberCount Number",
			"\tfor lines",
			"\t\tif StartsWith(\"has \")",
			"\t\t\t1");
		var textsType = typeParser.Members[0].Type;
		var testLines = new ValueInstance(textsType,
		[
			new ValueInstance("has logger"),
			new ValueInstance("Run"),
			new ValueInstance("\tbody")
		]);
		var instance = new ValueInstance(typeParser, [testLines]);
		var result = interpreter.Execute(
			typeParser.Methods.Single(m => m.Name == "MemberCount"), instance, []);
		Assert.That(result.Number, Is.EqualTo(1));
	}

	[Test]
	public void StrictTypeParserCountsMethods()
	{
		using var typeParser = CreateType(nameof(StrictTypeParserCountsMethods),
			"has lines Texts",
			"IsMethodHeader(line Text) Boolean",
			"\t(not line.StartsWith(\"has \")) and (not line.StartsWith(\"\\t\"))",
			"MethodCount Number",
			"\tfor lines",
			"\t\tif IsMethodHeader(value)",
			"\t\t\t1");
		var textsType = typeParser.Members[0].Type;
		var testLines = new ValueInstance(textsType,
		[
			new ValueInstance("has logger"),
			new ValueInstance("Run"),
			new ValueInstance("\tbody")
		]);
		var instance = new ValueInstance(typeParser, [testLines]);
		var result = interpreter.Execute(
			typeParser.Methods.Single(m => m.Name == "MethodCount"), instance, []);
		Assert.That(result.Number, Is.EqualTo(1));
	}

	[Test]
	public void StrictTypeParserExtractsMemberNames()
	{
		using var typeParser = CreateType(nameof(StrictTypeParserExtractsMemberNames),
			"has lines Texts",
			"ExtractMemberName(line Text) Text",
			"\tline.Substring(4, line.characters.Length - 4)",
			"MemberNames Texts",
			"\tfor lines",
			"\t\tif StartsWith(\"has \")",
			"\t\t\tExtractMemberName(value)");
		var textsType = typeParser.Members[0].Type;
		var testLines = new ValueInstance(textsType,
		[
			new ValueInstance("has logger"),
			new ValueInstance("Run"),
			new ValueInstance("\tbody")
		]);
		var instance = new ValueInstance(typeParser, [testLines]);
		var result = interpreter.Execute(
			typeParser.Methods.Single(m => m.Name == "MemberNames"), instance, []);
		Assert.That(result.List.Items, Has.Count.EqualTo(1));
		Assert.That(result.List.Items[0].Text, Is.EqualTo("logger"));
	}

	[Test]
	public void StrictTypeParserExtractsMethodHeaders()
	{
		using var typeParser = CreateType(nameof(StrictTypeParserExtractsMethodHeaders),
			"has lines Texts",
			"IsMethodHeader(line Text) Boolean",
			"\t(not line.StartsWith(\"has \")) and (not line.StartsWith(\"\\t\"))",
			"MethodHeaders Texts",
			"\tfor lines",
			"\t\tif IsMethodHeader(value)",
			"\t\t\tvalue");
		var textsType = typeParser.Members[0].Type;
		var testLines = new ValueInstance(textsType,
		[
			new ValueInstance("has logger"),
			new ValueInstance("Run"),
			new ValueInstance("\tbody")
		]);
		var instance = new ValueInstance(typeParser, [testLines]);
		var result = interpreter.Execute(
			typeParser.Methods.Single(m => m.Name == "MethodHeaders"), instance, []);
		Assert.That(result.List.Items, Has.Count.EqualTo(1));
		Assert.That(result.List.Items[0].Text, Is.EqualTo("Run"));
	}
}