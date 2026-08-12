namespace Strict.Language.Tests;

public sealed class TypeParserTests
{
	[SetUp]
	public void CreateParser() => parser = new MethodExpressionParser();

	private readonly Package package = TestPackage.Instance;
	public ExpressionParser parser = null!;

	[Test]
	public void EmptyLineIsNotAllowed() =>
		Assert.That(() => CreateType(nameof(EmptyLineIsNotAllowed), ""),
			Throws.InstanceOf<TypeParser.EmptyLineIsNotAllowed>());

	[Test]
	public void EmptyLineAtEndOfFileIsNotAllowed() =>
		Assert.That(
			() => CreateType(nameof(EmptyLineAtEndOfFileIsNotAllowed), "has logger", "Run", "\t5",
				""), Throws.InstanceOf<TypeParser.EmptyLineIsNotAllowed>());

	[Test]
	public void EmptyLineBetweenMethodsIsNotAllowed() =>
		Assert.That(
			() => CreateType(nameof(EmptyLineBetweenMethodsIsNotAllowed), "has logger", "Run", "\t5",
				"", "Other", "\t1"), Throws.InstanceOf<TypeParser.EmptyLineIsNotAllowed>());

	[Test]
	public void TrailingNewlineAtEndOfFileIsEmptyLineAndNotAllowed()
	{
		// File.ReadAllLines would drop this; Strict keeps "" so TypeParser rejects EOF newline.
		var lines = TypeLines.SplitLines("has logger\r\nRun\r\n\t5\r\n");
		Assert.That(lines, Is.EqualTo(new[] { "has logger", "Run", "\t5", "" }));
		Assert.That(
			() => new Type(package,
					new TypeLines(nameof(TrailingNewlineAtEndOfFileIsEmptyLineAndNotAllowed), lines)).
				ParseMembersAndMethods(parser),
			Throws.InstanceOf<TypeParser.EmptyLineIsNotAllowed>());
	}

	[Test]
	public void DoubleTrailingNewlineIsAlsoRejected()
	{
		var lines = TypeLines.SplitLines("has logger\nRun\n\t5\n\n");
		Assert.That(lines[^1], Is.EqualTo(""));
		Assert.That(
			() => new Type(package,
					new TypeLines(nameof(DoubleTrailingNewlineIsAlsoRejected), lines)).
				ParseMembersAndMethods(parser),
			Throws.InstanceOf<TypeParser.EmptyLineIsNotAllowed>());
	}

	[Test]
	public void SourceWithoutTrailingNewlineParses()
	{
		var lines = TypeLines.SplitLines("has logger\nRun\n\t5");
		Assert.That(lines, Is.EqualTo(new[] { "has logger", "Run", "\t5" }));
		using var type = new Type(package,
			new TypeLines(nameof(SourceWithoutTrailingNewlineParses), lines)).
			ParseMembersAndMethods(parser);
		Assert.That(type.Methods, Has.Count.EqualTo(1));
	}

	private void CreateType(string name, params string[] lines) =>
		new Type(package, new TypeLines(name, lines)).ParseMembersAndMethods(parser).Dispose();

	[Test]
	public void WhitespacesAreNotAllowed()
	{
		Assert.That(() => CreateType("Whitespace", " "),
			Throws.InstanceOf<TypeParser.ExtraWhitespacesFoundAtBeginningOfLine>());
		Assert.That(() => CreateType("ProgramWhitespace", " has App"),
			Throws.InstanceOf<TypeParser.ExtraWhitespacesFoundAtBeginningOfLine>());
		Assert.That(() => CreateType("TabWhitespace", "has\t"),
			Throws.InstanceOf<TypeParser.ExtraWhitespacesFoundAtEndOfLine>());
	}

	[Test]
	public void ExtraWhitespacesFoundAtBeginningOfLine() =>
		Assert.That(
			() => CreateType(nameof(ExtraWhitespacesFoundAtBeginningOfLine), "has logger", "Run",
				" constant a =5"), Throws.InstanceOf<TypeParser.ExtraWhitespacesFoundAtBeginningOfLine>());

	[TestCase("has any")]
	[TestCase("has random Any")]
	public void MemberWithTypeAnyIsNotAllowed(string line) =>
		Assert.That(() => CreateType(nameof(MemberWithTypeAnyIsNotAllowed) + line[5], line),
			Throws.InstanceOf<TypeParser.MemberWithTypeAnyIsNotAllowed>());

	[Test]
	public void MembersMustComeBeforeMethods() =>
		Assert.That(() => CreateType(nameof(MembersMustComeBeforeMethods), "Run", "has logger"),
			Throws.InstanceOf<TypeParser.MembersMustComeBeforeMethods>());

	[Test]
	public void MissingConstraintExpression() =>
		Assert.That(
			() => CreateType(nameof(MissingConstraintExpression),
				"mutable numbers with", "AddNumbers Number", "\tnumbers(0) + numbers(1)"),
			Throws.InstanceOf<TypeParser.MemberMissingConstraintExpression>());

	[Test]
	public void CurrentTypeCannotBeInstantiatedAsMemberType() =>
		Assert.That(
			() => CreateType(nameof(CurrentTypeCannotBeInstantiatedAsMemberType), "has number",
				"has currentType = CurrentTypeCannotBeInstantiatedAsMemberType(5)", "Unused", "\t1"),
			Throws.InstanceOf<TypeParser.CurrentTypeCannotBeInstantiatedAsMemberType>());

	[Test]
	public void TrivialEndlessSelfConstructionInFromIsDetected() =>
		Assert.That(
			() => CreateType(nameof(TrivialEndlessSelfConstructionInFromIsDetected),
				"has logger",
				"from(number)",
				$"\t{nameof(TrivialEndlessSelfConstructionInFromIsDetected)}(0)"),
			Throws.InstanceOf<TypeParser.TrivialEndlessSelfConstructionDetected>());

	[Test]
	public void SelfRecursiveCallWithSameArgumentsDirectCall()
	{
		var exception = Assert.Throws<TypeParser.SelfRecursiveCallWithSameArgumentsDetected>(() =>
			CreateType(nameof(SelfRecursiveCallWithSameArgumentsDirectCall),
				"has logger",
				"Foo(first Number, second Number)",
				"\tFoo(first, second)"));
		Assert.That(exception!.Message, Does.Contain("Foo(first Number, second Number)"));
		Assert.That(exception.Message, Does.Contain("arguments=(Number, Number)"));
	}

	[Test]
	public void SelfRecursiveCallWithSameArgumentsDotCall() =>
		Assert.That(
			() => CreateType(nameof(SelfRecursiveCallWithSameArgumentsDotCall),
				"has logger",
				"Bar(number)",
				"\tthis.Bar(number)"),
			Throws.InstanceOf<TypeParser.SelfRecursiveCallWithSameArgumentsDetected>());

	[Test]
	public void SelfRecursiveCallWithSameArgumentsTypeDotCall() =>
		Assert.That(
			() => CreateType(nameof(SelfRecursiveCallWithSameArgumentsTypeDotCall),
				"has logger",
				"Baz(number)",
				"\t" + nameof(SelfRecursiveCallWithSameArgumentsTypeDotCall) + ".Baz(number)"),
			Throws.InstanceOf<TypeParser.SelfRecursiveCallWithSameArgumentsDetected>());

	[Test]
	public void HugeConstantRangeIsDetected() =>
		Assert.That(
			() => CreateType(nameof(HugeConstantRangeIsDetected),
				"has logger",
				"Run",
				"\tRange(1,2000000001)"),
			Throws.InstanceOf<TypeParser.HugeConstantRangeNotAllowed>());

	[Test]
	public void RedundantReturnPreviousLineContainsValueAlready() =>
		Assert.That(
			() => CreateType(nameof(RedundantReturnPreviousLineContainsValueAlready), "has logger",
				"Run", "\tconstant number = 5", "\tnumber"),
			Throws.InstanceOf<TypeParser.RedundantReturnPreviousLineContainsValueAlready>());

	[Test]
	public void ReturnAsLastExpressionIsNotNeeded() =>
		Assert.That(
			() => CreateType(nameof(ReturnAsLastExpressionIsNotNeeded), "has logger",
				"Run", "\treturn true"),
			Throws.InstanceOf<Body.ReturnAsLastExpressionIsNotNeeded>());

	[Test]
	public void IsMemberDefaultsToBooleanType()
	{
		using var simpleType =
			new Type(package,
				new TypeLines(nameof(IsMemberDefaultsToBooleanType), "has isDefault", "has IsConstant",
					"Run", "\tisDefault and IsConstant")).ParseMembersAndMethods(parser);
		Assert.That(simpleType.Members[0].Type,
			Is.EqualTo(TestPackage.Instance.GetType(Type.Boolean)));
		Assert.That(simpleType.Members[1].Type,
			Is.EqualTo(TestPackage.Instance.GetType(Type.Boolean)));
	}

	[Test]
	public void AutoNumberedEnumConstantDoesNotWrapExistingTwoArgType()
	{
		using var localPackage = new Package(package, "EnumAddConst");
		using var add = new Type(localPackage,
			new TypeLines("Add", "has first Number", "has second Number",
				"from(first Number, second Number)", "\tfirst + second")).
			ParseMembersAndMethods(parser);
		using var instruction = new Type(localPackage,
			new TypeLines("InstructionAddConst",
				"constant StoreSeparator = 10", "constant Add")).ParseMembersAndMethods(parser);
		var addConstant = instruction.Members.Single(member => member.Name == "Add");
		Assert.That(addConstant.Type.Name, Is.EqualTo(Type.Number));
		Assert.That(addConstant.InitialValue?.ToString(), Is.EqualTo("11"));
	}
}