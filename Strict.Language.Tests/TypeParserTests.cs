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
	public void SelfRecursiveCallWithSameArgumentsDirectCall() =>
		Assert.That(
			() => CreateType(nameof(SelfRecursiveCallWithSameArgumentsDirectCall),
				"has logger",
				"Foo(first Number, second Number)",
				"\tFoo(first, second)"),
			Throws.InstanceOf<TypeParser.SelfRecursiveCallWithSameArgumentsDetected>());

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
}