namespace Strict.Language.Tests;

public sealed class TypeParserTests
{
	[SetUp]
	public void CreateParser() => parser = new MethodExpressionParser();

	private readonly Package package = TestPackage.Instance;
	public ExpressionParser parser = null!;

	[Test]
	public void EmptyLineIsNotAllowed() =>
		Assert.That(() =>
			{
				using var _ = CreateType(nameof(EmptyLineIsNotAllowed), "");
			}, //ncrunch: no coverage
			Throws.InstanceOf<TypeParser.EmptyLineIsNotAllowed>().With.Message.Contains("line1"));

	private Type CreateType(string name, params string[] lines) =>
		new Type(package, new TypeLines(name, lines)).ParseMembersAndMethods(parser);

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
			() =>
			{
				var typeName = nameof(TrivialEndlessSelfConstructionInFromIsDetected);
				CreateType(typeName,
					$"from(n Number)",
					$"\t{typeName}(0)");
			},
			Throws.InstanceOf<TypeParser.TrivialEndlessSelfConstructionDetected>());

	[Test]
	public void SelfRecursiveCallWithSameArgumentsIsDetectedForDirectCall() =>
		Assert.That(
			() => CreateType(nameof(SelfRecursiveCallWithSameArgumentsIsDetectedForDirectCall),
				"Foo(a Number, b Number)",
				"\tFoo(a, b)"),
			Throws.InstanceOf<TypeParser.SelfRecursiveCallWithSameArgumentsDetected>());

	[Test]
	public void SelfRecursiveCallWithSameArgumentsIsDetectedForThisDotCall() =>
		Assert.That(
			() => CreateType(nameof(SelfRecursiveCallWithSameArgumentsIsDetectedForThisDotCall),
				"Bar(x Number)",
				"\tthis.Bar(x)"),
			Throws.InstanceOf<TypeParser.SelfRecursiveCallWithSameArgumentsDetected>());

	[Test]
	public void SelfRecursiveCallWithSameArgumentsIsDetectedForTypeDotCall() =>
		Assert.That(
			() => CreateType(nameof(SelfRecursiveCallWithSameArgumentsIsDetectedForTypeDotCall),
				"Baz(y Number)",
				"\tSelfRecursiveCallWithSameArgumentsIsDetectedForTypeDotCall.Baz(y)"),
			Throws.InstanceOf<TypeParser.SelfRecursiveCallWithSameArgumentsDetected>());

	[Test]
	public void HugeConstantRangeIsDetected() =>
		Assert.That(
			() => CreateType(nameof(HugeConstantRangeIsDetected),
				"Run",
				"\tRange(1,2000000001)"),
			Throws.InstanceOf<TypeParser.HugeConstantRangeNotAllowed>());
}