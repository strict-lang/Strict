using NUnit.Framework;

namespace Strict.Language.Tests;

public sealed class BracketParserTests
{
	[Test]
	public void InvalidClosingBracket() =>
		Assert.That(() => new BracketParser(")(").Groups,
			Throws.InstanceOf<BracketParser.UnbalancedBracketsFound>()!);

	[TestCase(")(")]
	[TestCase("(2 + 2)(")]
	[TestCase("(((2 + 2))")]
	public void UnbalancedBracketsFound(string code) =>
		Assert.That(() => new BracketParser(code).Groups,
			Throws.InstanceOf<BracketParser.UnbalancedBracketsFound>()!);

	[TestCase("")]
	[TestCase("1")]
	[TestCase("1 + 2")]
	//[TestCase("(1, 2)")]
	public void StringWithoutGroups(string code) => Assert.That(new BracketParser(code).Groups, Is.Empty);

	[Test]
	public void EmptyGroup()
	{
		var groups = new BracketParser("()").Groups;
		Assert.That(groups, Has.Count.EqualTo(1));
		Assert.That(groups[0], Is.EqualTo(new Group(1)));
	}

	[Test]
	public void SingleGroup()
	{
		var groups = new BracketParser("(1 + 2)").Groups;
		Assert.That(groups, Has.Count.EqualTo(1));
		Assert.That(groups[0], Is.EqualTo(new Group(1) { Length = "1 + 2".Length }));
	}

	[Test]
	public void MultipleGroups()
	{
		var groups = new BracketParser("(1 + 2) * (3 + 5)").Groups;
		Assert.That(groups, Has.Count!.EqualTo(2));
		Assert.That(groups[0], Is.EqualTo(new Group(1) { Length = "1 + 2".Length }));
		Assert.That(groups[1], Is.EqualTo(new Group(11) { Length = "3 + 5".Length }));
	}

	[Test]
	public void NestedGroups()
	{
		var groups = new BracketParser("(((1 + 2) * (3 + 5)) / (6 + 3))").Groups;
		Assert.That(groups, Has.Count!.EqualTo(5));
		Assert.That(groups[0], Is.EqualTo(new Group(3) { Length = "1 + 2".Length }));
		Assert.That(groups[1], Is.EqualTo(new Group(13) { Length = "3 + 5".Length }));
		Assert.That(groups[2],
			Is.EqualTo(new Group(2) { Length = "(1 + 2) * (3 + 5)".Length }));
		Assert.That(groups[3], Is.EqualTo(new Group(24) { Length = "6 + 3".Length }));
		Assert.That(groups[4],
			Is.EqualTo(new Group(1) { Length = "((1 + 2) * (3 + 5)) / (6 + 3)".Length }));
	}
}