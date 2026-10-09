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
	public void ListCallOnMemberInsideMemberChain()
	{
		using var type = new Type(TestPackage.Instance, new TypeLines(nameof(ListCallOnMemberInsideMemberChain),
			"has texts", $"SecondLength(other {nameof(ListCallOnMemberInsideMemberChain)}) Number",
			"\tother.texts(1).Length")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(type.Methods[0].GetBodyAndParseIfNeeded().ToString(),
			Is.EqualTo("other.texts(1).Length"));
	}
}