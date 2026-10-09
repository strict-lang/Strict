using Strict.Language.Tests;

namespace Strict.Expressions.Tests;

public sealed class ListCallTests : TestExpressions
{
	[TestCase("constant numbers = (1, 2, 3)", "numbers(0)")]
	[TestCase("constant texts = (\"something\", \"someOtherThing\")", "texts(1)")]
	public void ListCallToString(params string[] lines) =>
		Assert.That(ParseExpression(lines).ToString(),
			Is.EqualTo(string.Join(Environment.NewLine, lines)));

	[Test]
	public void RangeLoopValueIsTheCurrentNumber()
	{
		using var type = new Type(TestPackage.Instance, new TypeLines(
			nameof(RangeLoopValueIsTheCurrentNumber), "has texts", "Doubled Numbers",
			"\tfor Range(2, texts.Length)", "\t\tvalue.Floor")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(() => type.Methods[0].GetBodyAndParseIfNeeded(), Throws.Nothing);
	}

	[Test]
	public void MutableReassignmentInsideLoopKeepsValueType()
	{
		using var type = new Type(TestPackage.Instance, new TypeLines(
			nameof(MutableReassignmentInsideLoopKeepsValueType), "has texts", "Lengths Number",
			"\tmutable sum = 0", "\tmutable names = List(Mutable(Text))", "\tmutable isCounting = false", "\tfor texts",
			"\t\tif value is \"a\"", "\t\t\tisCounting = true", "\t\tif isCounting",
			"\t\t\tnames.Add(value)", "\t\tsum = sum + value.IndexOf(\"a\")", "\tsum + names.Length")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(() => type.Methods[0].GetBodyAndParseIfNeeded(), Throws.Nothing);
	}

	[Test]
	public void ListCallOnMemberInsideMemberChain()
	{
		using var type = new Type(TestPackage.Instance, new TypeLines(nameof(ListCallOnMemberInsideMemberChain),
			"has texts", $"SecondLength(other {nameof(ListCallOnMemberInsideMemberChain)}) Number",
			"\tother.texts(1).Length")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(type.Methods[0].GetBodyAndParseIfNeeded().ToString(),
			Is.EqualTo("other.texts(1).Length"));
	}
}