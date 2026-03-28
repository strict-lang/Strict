using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime.Tests;

public sealed class InterpreterTests
{
	[SetUp]
	public void CreateExecutor() =>
		interpreter = new Interpreter(TestPackage.Instance, TestBehavior.Disabled);

	private Interpreter interpreter = null!;

	[Test]
	public void MissingArgument()
	{
		using var t = CreateCalcType();
		var method = t.Methods.Single(m => m.Name == "Add");
		Assert.That(() => interpreter.Execute(method),
			Throws.TypeOf<Interpreter.MissingArgument>().With.Message.StartsWith("first"));
	}

	[Test]
	public void FromConstructorWithExistingInstanceThrows()
	{
		using var t = CreateType(nameof(FromConstructorWithExistingInstanceThrows), "has number",
			"from(number Number)", "\tvalue");
		var method = t.Methods.Single(m => m.Name == Method.From);
		var number = new ValueInstance(interpreter.numberType, 3);
		var instance = new ValueInstance(t, 1);
		Assert.That(() => interpreter.Execute(method, instance, [number]),
			Throws.InstanceOf<MethodCall.CannotCallFromConstructorWithExistingInstance>());
	}

	private static Type CreateCalcType() =>
		CreateType("Calc", "mutable last Number", "Add(first Number, second = 1) Number",
			"\tAdd(1) is 2", "\tlast = first + second");

	private static Type CreateType(string name, params string[] lines) =>
		new Type(TestPackage.Instance, new TypeLines(name, lines)).ParseMembersAndMethods(
			new MethodExpressionParser());

	[Test]
	public void UseDefaultValue()
	{
		using var t = CreateCalcType();
		var method = t.Methods.Single(m => m.Name == "Add");
		var result = interpreter.Execute(method, interpreter.noneInstance,
			[new ValueInstance(interpreter.numberType, 5)]);
		Assert.That(result.Number, Is.EqualTo(6));
	}

	[Test]
	public void TooManyArguments()
	{
		using var t = CreateCalcType();
		var method = t.Methods.Single(m => m.Name == "Add");
		Assert.That(() => interpreter.Execute(method, interpreter.noneInstance, [
			new ValueInstance(interpreter.numberType, 1),
			new ValueInstance(interpreter.numberType, 2),
			new ValueInstance(interpreter.numberType, 3)
		]), Throws.InstanceOf<Interpreter.TooManyArguments>().With.Message.StartsWith("Number: 3"));
	}

	[Test]
	public void ArgumentDoesNotMapToMethodParameters()
	{
		using var t = CreateType(nameof(ArgumentDoesNotMapToMethodParameters), "has number",
			"Use(number Number) Number", "\tnumber");
		var method = t.Methods.Single(m => m.Name == "Use");
		Assert.That(() => interpreter.Execute(method, interpreter.noneInstance, [interpreter.trueInstance]),
			Throws.InstanceOf<Interpreter.ArgumentDoesNotMapToMethodParameters>());
	}

	[Test]
	public void EvaluateValueAndVariableAndParameterCalls()
	{
		using var t = CreateCalcType();
		var method = t.Methods.Single(m => m.Name == "Add");
		var first = new ValueInstance(interpreter.numberType, 5);
		var second = new ValueInstance(interpreter.numberType, 7);
		var result = interpreter.Execute(method, interpreter.noneInstance, [first, second]);
		Assert.That(result.Number, Is.EqualTo(12));
	}

	[Test]
	public void EvaluateDeclaration()
	{
		using var t = CreateType(nameof(EvaluateDeclaration), "mutable last Number",
			"AddFive(number) Number", "\tconstant five = 5", "\tnumber + five");
		var method = t.Methods.Single(m => m.Name == "AddFive");
		var number = new ValueInstance(interpreter.numberType, 5);
		var result = interpreter.Execute(method, interpreter.noneInstance, [number]);
		Assert.That(result.Number, Is.EqualTo(10));
	}

	[Test]
	public void EvaluateAllArithmeticOperators()
	{
		Assert.That(interpreter.Execute(GetBinaryOperator(BinaryOperator.Plus), N(2), [N(3)]).Number,
			Is.EqualTo(5));
		Assert.That(interpreter.Execute(GetBinaryOperator(BinaryOperator.Minus), N(8), [N(3)]).Number,
			Is.EqualTo(5));
		Assert.That(interpreter.Execute(GetBinaryOperator(BinaryOperator.Multiply), N(6), [N(7)]).Number,
			Is.EqualTo(42));
		Assert.That(interpreter.Execute(GetBinaryOperator(BinaryOperator.Divide), N(8), [N(2)]).Number,
			Is.EqualTo(4));
		Assert.That(interpreter.Execute(GetBinaryOperator(BinaryOperator.Modulate), N(8), [N(3)]).Number,
			Is.EqualTo(2));
		Assert.That(interpreter.Execute(GetBinaryOperator(BinaryOperator.Power), N(2), [N(3)]).Number,
			Is.EqualTo(8));
	}

	private Method GetBinaryOperator(string op) =>
		numberType.Methods.Single(m => m.Name == op && m.Parameters.Count == 1);

	private readonly Type numberType = TestPackage.Instance.GetType(Type.Number);
	private ValueInstance N(double x) => new(numberType, x);

	[Test]
	public void AddTwoTexts()
	{
		var text = TestPackage.Instance.GetType(Type.Text);
		var plusText = text.Methods.Single(m =>
			m.Name == BinaryOperator.Plus && m.Parameters is [{ Type.IsText: true }]);
		var result = interpreter.Execute(plusText, new ValueInstance("hi "), [new ValueInstance("there")]);
		Assert.That(result.Text, Is.EqualTo("hi there"));
	}

  [Test]
	public async Task TextInFindsMatchInsideText()
	{
		using var strict = await new Repositories(new MethodExpressionParser()).LoadStrictPackage();
    var inMethod = strict.GetType(Type.Text).Methods.Single(m => m.Name == "in");
    var result = new Interpreter(strict, TestBehavior.Disabled).Execute(inMethod,
			new ValueInstance("hello there"), [new ValueInstance("lo")]);
		Assert.That(result.Boolean, Is.True);
	}

	[Test]
	public async Task TextCombineSupportsCharacterSeparator()
	{
		using var strict = await new Repositories(new MethodExpressionParser()).LoadStrictPackage();
		var text = strict.GetType(Type.Text);
		var combineMethod = text.Methods.Single(method => method.Name == "Combine");
		var interpreterForStrict = new Interpreter(strict, TestBehavior.Disabled);
		var texts = strict.GetListImplementationType(text);
		var character = strict.GetType(Type.Character);
		var result = interpreterForStrict.Execute(combineMethod, interpreterForStrict.noneInstance,
		[
			new ValueInstance(texts, [new ValueInstance("hi"), new ValueInstance("there")]),
			new ValueInstance(character, 10)
		]);
		Assert.That(result.Text, Is.EqualTo("hi\nthere"));
	}

	[Test]
	public async Task TextSplitWorksWithSeparatorParameter()
	{
		using var strict = await new Repositories(new MethodExpressionParser()).LoadStrictPackage();
		var splitMethod = strict.GetType(Type.Text).Methods.Single(method => method.Name == "Split");
		var interpreterForStrict = new Interpreter(strict);
		var result = interpreterForStrict.Execute(splitMethod, new ValueInstance("a,b"),
			[new ValueInstance(",")]);
		Assert.That(result.List.Items.Select(item => item.Text), Is.EqualTo(new[] { "a", "b" }));
	}

	[Test]
	public void StringLiteralBackslashNParsesAsNewLine()
	{
		using var type = CreateType(nameof(StringLiteralBackslashNParsesAsNewLine), "has number",
			"GetText Text", "\t\"hi\\nthere\"");
		var result = interpreter.Execute(type.Methods.Single(method => method.Name == "GetText"),
			interpreter.noneInstance, []);
		Assert.That(result.Text, Is.EqualTo("hi\nthere"));
	}

	[Test]
	public void AddTwoCharactersReturnsConcatenatedText()
	{
		var characterType = TestPackage.Instance.GetType(Type.Character);
		var plus = characterType.Methods.Single(method => method.Name == BinaryOperator.Plus);
		var result = interpreter.Execute(plus, new ValueInstance(characterType, '1'),
			[new ValueInstance(characterType, '2')]);
		Assert.That(result.Text, Is.EqualTo("12"));
	}

	[Test]
	public void EvaluateAllComparisonOperators()
	{
		Assert.That(interpreter.Execute(GetBinaryOperator(BinaryOperator.Greater), N(5), [N(3)]),
			Is.EqualTo(interpreter.trueInstance));
		Assert.That(interpreter.Execute(GetBinaryOperator(BinaryOperator.Smaller), N(2), [N(3)]),
			Is.EqualTo(interpreter.trueInstance));
		Assert.That(interpreter.Execute(GetBinaryOperator(BinaryOperator.GreaterOrEqual), N(5), [N(3)]),
			Is.EqualTo(interpreter.trueInstance));
		Assert.That(interpreter.Execute(GetBinaryOperator(BinaryOperator.GreaterOrEqual), N(5), [N(5)]),
			Is.EqualTo(interpreter.trueInstance));
		Assert.That(interpreter.Execute(GetBinaryOperator(BinaryOperator.SmallerOrEqual), N(2), [N(3)]),
			Is.EqualTo(interpreter.trueInstance));
		Assert.That(interpreter.Execute(GetBinaryOperator(BinaryOperator.SmallerOrEqual), N(3), [N(3)]),
			Is.EqualTo(interpreter.trueInstance));
		Assert.That(interpreter.Execute(GetBinaryOperator(BinaryOperator.Is), N(3), [N(3)]),
			Is.EqualTo(interpreter.trueInstance));
		Assert.That(interpreter.Execute(GetBinaryOperator(BinaryOperator.Is), N(3), [N(4)]),
			Is.EqualTo(interpreter.falseInstance));
	}

	[Test]
	public void EvaluateAllLogicalOperators()
	{
		var not = booleanType.Methods.Single(m =>
			m.Name == UnaryOperator.Not && m.ReturnType.IsBoolean && m.Parameters.Count == 0);
		var and = GetBinaryBooleanOperator(BinaryOperator.And);
		AssertBooleanOperation(and, true, true, true);
		AssertBooleanOperation(and, true, false, false);
		var or = GetBinaryBooleanOperator(BinaryOperator.Or);
		AssertBooleanOperation(or, true, false, false);
		AssertBooleanOperation(or, false, false, true);
		var xor = GetBinaryBooleanOperator(BinaryOperator.Xor);
		AssertBooleanOperation(xor, true, false, true);
		AssertBooleanOperation(xor, true, true, false);
		Assert.That(interpreter.Execute(not, interpreter.falseInstance, []), Is.EqualTo(interpreter.trueInstance));
	}

	private readonly Type booleanType = TestPackage.Instance.GetType(Type.Boolean);

	private Method GetBinaryBooleanOperator(string op) =>
		booleanType.Methods.Single(m =>
			m.Name == op && m.ReturnType.IsBoolean && m.Parameters is [{ Type.IsBoolean: true }]);

	private void AssertBooleanOperation(Method method, bool first, bool second, bool result) =>
		Assert.That(interpreter.Execute(method, interpreter.ToBoolean(first), [interpreter.ToBoolean(second)]),
			Is.EqualTo(interpreter.ToBoolean(result)));

	[Test]
	public void EvaluateMemberCallFromStaticConstant()
	{
		using var t = CreateType(nameof(EvaluateMemberCallFromStaticConstant), "mutable last Number",
			"GetTab Character", "\tCharacter.Tab");
		var result = interpreter.Execute(t.Methods.Single(m => m.Name == "GetTab"));
		Assert.That(result.IsPrimitiveType(interpreter.characterType), Is.True, result.ToString());
		Assert.That(result.Number, Is.EqualTo(7));
	}

	[Test]
	public void EvaluateBooleanComparisons()
	{
		using var t = CreateType(nameof(EvaluateBooleanComparisons), "mutable last Boolean",
			"IfDifferent Boolean", "\tlast is false");
		Assert.That(
			interpreter.Execute(t.Methods.Single(m => m.Name == "IfDifferent"),
				new ValueInstance(t, [interpreter.falseInstance]), []),
			Is.EqualTo(interpreter.trueInstance));
	}

	[Test]
	public void EvaluateRangeEquality()
	{
		using var t = CreateType(nameof(EvaluateRangeEquality), "has number", "Compare Boolean",
			"\tRange(0, 5) is Range(0, 5)");
		Assert.That(interpreter.Execute(t.Methods.Single(m => m.Name == "Compare"), interpreter.noneInstance,
			[]), Is.EqualTo(interpreter.trueInstance));
	}

	[Test]
	public void MultilineMethodRequiresTests()
	{
		using var t = CreateType(nameof(MultilineMethodRequiresTests), "has number", "GetText Text",
			"\tif number is 0", "\t\treturn \"\"", "\tnumber to Text");
		var instance = new ValueInstance(t,
			[new ValueInstance(interpreter.numberType, 5.0)]);
		Assert.That(interpreter.Execute(t.Methods.Single(m => m.Name == "GetText"), instance, []).Text,
			Is.EqualTo("5"));
	}

	[Test]
	public void MethodWithoutTestsThrowsMethodRequiresTestDuringValidation()
	{
		using var t = CreateType("NoTestsNeedValidation", "has number", "Compute Number", "\tif true",
			"\t\treturn 1", "\t2");
		var method = t.Methods.Single(m => m.Name == "Compute");
		var validatingExecutor = new Interpreter(TestPackage.Instance);
		var ex = Assert.Throws<InterpreterExecutionFailed>(() => validatingExecutor.Execute(method));
		Assert.That(ex!.InnerException, Is.InstanceOf<Interpreter.MethodRequiresTest>());
	}

	[Test]
	public void MethodRequiresTestWhenParsingFailsDuringValidation()
	{
		using var t = CreateType(nameof(MethodRequiresTestWhenParsingFailsDuringValidation),
			"has number", "Compute Number", "\tunknown(1)");
		var method = t.Methods.Single(m => m.Name == "Compute");
		var validatingExecutor = new Interpreter(TestPackage.Instance);
		Assert.That(() => validatingExecutor.Execute(method),
			Throws.InstanceOf<Interpreter.MethodRequiresTest>());
	}

	[Test]
	public void InvalidTypeForFromConstructor()
	{
		using var t = CreateType(nameof(InvalidTypeForFromConstructor), "has flag Boolean",
			"from(flag Boolean, other Boolean)", "\tvalue");
		var method = t.Methods.Single(m => m.Name == Method.From);
		var number = new ValueInstance(interpreter.numberType, 1.0);
		var boolean = interpreter.trueInstance;
		Assert.That(() => interpreter.Execute(method, interpreter.noneInstance, [number, boolean]),
			Throws.InstanceOf<Interpreter.InvalidTypeForArgument>());
	}

	[Test]
	public void FromConstructorConvertsSingleCharText()
	{
		using var t = CreateType(nameof(FromConstructorConvertsSingleCharText), "has number",
			"has text", "from(number Number, text Text)", "\tvalue");
		var method = t.Methods.Single(m => m.Name == Method.From);
		var numberText = new ValueInstance("A");
		var text = new ValueInstance("ok");
		var result = interpreter.Execute(method, interpreter.noneInstance, [numberText, text]);
		var members = result.TryGetValueTypeInstance()!;
		Assert.That(members["number"].Number, Is.EqualTo(65));
	}

	[Test]
	public void CompareNumberToText()
	{
		using var t = CreateType(nameof(CompareNumberToText), "has number", "Compare",
			"\t\"5\" is 5");
		Assert.That(
			interpreter.Execute(t.Methods.Single(m => m.Name == "Compare"), interpreter.noneInstance, []),
			Is.EqualTo(interpreter.trueInstance));
	}

	[Test]
	public void CompareTextToCharacterTab()
	{
		using var t = CreateType(nameof(CompareTextToCharacterTab), "has number", "Compare Boolean",
			"\t\"7\" is Character.Tab");
		Assert.That(
			interpreter.Execute(t.Methods.Single(m => m.Name == "Compare"), interpreter.noneInstance, []),
			Is.EqualTo(interpreter.falseInstance));
	}

	[Test]
	public void StackOverflowCallingYourselfWithSameArguments() =>
		Assert.That(() =>
			{
				using var t = CreateType(nameof(StackOverflowCallingYourselfWithSameArguments),
					"has number", "Recursive(other Number)", "\tRecursive(other)");
			}, //ncrunch: no coverage
			Throws.InstanceOf<TypeParser.SelfRecursiveCallWithSameArgumentsDetected>());

	[Test]
	public void StackOverflowCallingYourselfWithSameInstanceMember()
	{
		using var t = CreateType(nameof(StackOverflowCallingYourselfWithSameInstanceMember),
			"has number", "Recursive(other Number)", "\tRecursive(number)");
		Assert.That(() => interpreter.Execute(t.Methods.Single(m => m.Name == "Recursive"),
				new ValueInstance(t, [new ValueInstance(interpreter.numberType, 3.0)]),
				[new ValueInstance(interpreter.numberType, 1.0)]),
			Throws.InstanceOf<Interpreter.StackOverflowCallingItselfWithSameInstanceAndArguments>());
	}

	[Test]
	public void StackOverflowMessageShowsCallDetails()
	{
		using var t = CreateType(nameof(StackOverflowMessageShowsCallDetails),
			"has number", "Recursive(other Number)", "\tRecursive(number)");
		var exception = Assert.Throws<Interpreter.StackOverflowCallingItselfWithSameInstanceAndArguments>(() =>
			interpreter.Execute(t.Methods.Single(m => m.Name == "Recursive"),
				new ValueInstance(t, [new ValueInstance(interpreter.numberType, 3.0)]),
				[new ValueInstance(interpreter.numberType, 1.0)]));
		Assert.That(exception!.Message, Does.Contain("Recursive(other Number)"));
		Assert.That(exception.Message, Does.Contain("instance="));
		Assert.That(exception.Message, Does.Contain("arguments=(Number: 1)"));
	}

	[Test]
	public void StackOverflowDetectionChecksGrandParentContextToo()
	{
		using var t = CreateType(nameof(StackOverflowDetectionChecksGrandParentContextToo),
			"has number",
			"Repeat(other Number) Number",
			"\tother",
			"Other(other Number) Number",
			"\tother");
		var repeat = t.Methods.Single(m => m.Name == "Repeat");
		var other = t.Methods.Single(m => m.Name == "Other");
		var instance = new ValueInstance(t, [new ValueInstance(interpreter.numberType, 3.0)]);
		var argument = new ValueInstance(interpreter.numberType, 1.0);
		var grandParent =
			new ExecutionContext(t, repeat, instance) { Variables = { ["other"] = argument } };
		var parent = new ExecutionContext(t, other, instance, grandParent);
		Assert.That(() => interpreter.Execute(repeat, instance, [argument], parent),
			Throws.InstanceOf<Interpreter.StackOverflowCallingItselfWithSameInstanceAndArguments>());
	}

	[Test]
	public void TextPlusOperatorUsesListCombineWithoutRecursiveFallback()
	{
		var textType = TestPackage.Instance.GetType(Type.Text);
		var plus = textType.Methods.Single(method => method.Name == BinaryOperator.Plus &&
			method.Parameters.Count == 1 && method.Parameters[0].Type.IsText);
		var result = interpreter.Execute(plus, new ValueInstance("Hey"), [new ValueInstance(" you")]);
		Assert.That(result.Text, Is.EqualTo("Hey you"));
	}

	[Test]
	public void CallNumberPlusOperator()
	{
		using var t = CreateType(nameof(CallNumberPlusOperator), "has number", "+(text) Number",
			"\tnumber + text.Length");
		Assert.That(interpreter.Execute(t.Methods.Single(m => m.Name == BinaryOperator.Plus),
			new ValueInstance(t, [new ValueInstance(interpreter.numberType, 5.0)]),
			[new ValueInstance("abc")]).Number, Is.EqualTo(5 + 3));
	}

	[Test]
	public void InlineListReferencingMember()
	{
		using var t = CreateType(nameof(InlineListReferencingMember),
			"has first Number", "has second Number", "GetCount Number",
			"\t(1, 2, 3).Length is 3",
			"\tlet myList = (second, 2, 3)",
			"\tmyList.Length");
		var validatingExecutor = new Interpreter(TestPackage.Instance);
		Assert.That(
			validatingExecutor.Execute(t.Methods.Single(m => m.Name == "GetCount")).Number, Is.EqualTo(3));
	}

	[Test]
	public void InlineDictionaryDeclarationLength()
	{
		using var t = CreateType(nameof(InlineDictionaryDeclarationLength),
			"has number", "GetCount Number",
			"\t(1, 2, 3).Length is 3",
			"\tconstant myDict = Dictionary(Number, Number)",
			"\tmyDict.Length");
		var validatingExecutor = new Interpreter(TestPackage.Instance);
		Assert.That(
			validatingExecutor.Execute(t.Methods.Single(m => m.Name == "GetCount")).Number, Is.EqualTo(0));
	}

	[Test]
	public void InlineTestDictionaryDeclaration()
	{
		using var t = CreateType(nameof(InlineTestDictionaryDeclaration),
			"has number", "Run Number",
			"\tconstant myDict = Dictionary(Text, Text)",
			"\tmyDict.Length is 0",
			"\tnumber");
		var validatingExecutor = new Interpreter(TestPackage.Instance);
		Assert.That(
			validatingExecutor.Execute(t.Methods.Single(m => m.Name == "Run")).Number, Is.EqualTo(0));
	}

	[Test]
	public void CannotCallMethodWithWrongInstanceThrows()
	{
		using var t = CreateType(nameof(CannotCallMethodWithWrongInstanceThrows), "has number",
			"Compute Number", "\tnumber");
		var method = t.Methods.Single(m => m.Name == "Compute");
		Assert.That(() => interpreter.Execute(method, interpreter.trueInstance, []),
			Throws.InstanceOf<Interpreter.CannotCallMethodWithWrongInstance>());
	}

	[Test]
	public void MutableDeclarationWithMutableValueTracksStatistics()
	{
		using var t = CreateType(nameof(MutableDeclarationWithMutableValueTracksStatistics),
			"has number", "Run Number",
			"\tmutable vx = number",
			"\tmutable vy = vx",
			"\tvx + vy");
		interpreter.Execute(t.Methods.Single(m => m.Name == "Run"), interpreter.noneInstance, []);
		Assert.That(interpreter.Statistics.MutableDeclarationCount, Is.EqualTo(1));
		Assert.That(interpreter.Statistics.MutableUsageCount, Is.EqualTo(1));
	}

	[Test]
	public void IsNotOperatorReturnsTrueForDifferentValues()
	{
		using var t = CreateType(nameof(IsNotOperatorReturnsTrueForDifferentValues), "has number",
			"Check Boolean", "\t1 is not 2");
		Assert.That(
			interpreter.Execute(t.Methods.Single(m => m.Name == "Check"), interpreter.noneInstance, []),
			Is.EqualTo(interpreter.trueInstance));
	}

	[Test]
	public void IsNotOperatorReturnsFalseForSameValues()
	{
		using var t = CreateType(nameof(IsNotOperatorReturnsFalseForSameValues), "has number",
			"Check Boolean", "\t1 is not 1");
		Assert.That(
			interpreter.Execute(t.Methods.Single(m => m.Name == "Check"), interpreter.noneInstance, []),
			Is.EqualTo(interpreter.falseInstance));
	}

	[Test]
	public void IsNotErrorReturnsFalseWhenBothAreErrors()
	{
		using var t = CreateType(nameof(IsNotErrorReturnsFalseWhenBothAreErrors), "has number",
			"Check Boolean",
			"\tconstant err = Error(\"test\")",
			"\terr is not err");
		Assert.That(
			interpreter.Execute(t.Methods.Single(m => m.Name == "Check"), interpreter.noneInstance, []),
			Is.EqualTo(interpreter.falseInstance));
	}

	[Test]
	public void ExecuteRunMethod() =>
		interpreter.ExecuteRunMethod(CreateType(nameof(ExecuteRunMethod), "has number", "Run", "\tnumber"));

	[Test]
	public void ExecuteRunMethodWillFailIfThereIsNoRunMethod() =>
		Assert.That(() => interpreter.ExecuteRunMethod(
			CreateType(nameof(ExecuteRunMethodWillFailIfThereIsNoRunMethod), "has number",
				"GetNumber Number", "\tnumber")), Throws.InstanceOf<Interpreter.MethodNotFound>());

	[Test]
	public void ComputeNumber()
	{
		using var t = CreateType(nameof(ComputeNumber), "has celsius Number",
			"ConvertToFahrenheit Number",
			"\tcelsius * 9 / 5 + 32");
		Assert.That(
			interpreter.Execute(t.Methods.Single(m => m.Name == "ConvertToFahrenheit"),
				new ValueInstance(t, 100), []).Number,
			Is.EqualTo(100 * 9 / 5 + 32));
	}

	[Test]
	public void ArithmeticFallbackErrorShowsMethodAndCallerContext()
	{
		using var type = CreateType(nameof(ArithmeticFallbackErrorShowsMethodAndCallerContext),
			"has values Texts",
			"Combine(separator Text) Text",
			"\tfor values",
      "\t\tvalue + (\"b\")",
			"Run Text",
			"\tCombine(\"a\")");
		var runMethod = type.Methods.Single(method => method.Name == "Run");
		var texts = type.GetListImplementationType(type.GetType(Type.Text));
		var instance = new ValueInstance(type,
      [new ValueInstance(texts, [new ValueInstance("x")])]);
    var exception = Assert.Throws<InterpreterExecutionFailed>(() =>
			interpreter.Execute(runMethod, instance, []));
		Assert.That(exception!.Message,
			Does.Contain("Arithmetic fallback is not allowed for core type Text operator +"));
    Assert.That(exception.Message, Does.Contain("method=Combine("));
		Assert.That(exception.Message, Does.Contain("call=value + (\"b\")"));
		Assert.That(exception.Message, Does.Contain("Run Text"));
   Assert.That(exception.Message, Does.Contain(":line "));
	}
}