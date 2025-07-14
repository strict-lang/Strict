using NUnit.Framework;
using static Strict.Language.NamedType;
using Strict.Expressions;

namespace Strict.Language.Tests;

public class KeywordTests
{
	[SetUp]
	public void CreatePackageAndType()
	{
		package = new TestPackage();
		parser = new MethodExpressionParser();
	}

	private Package package = null!;
	private ExpressionParser parser = null!;

	[TestCaseSource(nameof(KeywordsList))]
	public void CannotUseKeywordsAsMemberName(string name) =>
		Assert.That(() => new Type(package,
			new TypeLines(name + nameof(CannotUseKeywordsAsMemberName),
				$"has {name} Number",
				"Run",
				"\t5")).ParseMembersAndMethods(new MethodExpressionParser()), Throws.InstanceOf<ParsingFailed>().With.InnerException.InstanceOf<CannotUseKeywordsAsName>().With.Message.Contains($"{name} is a keyword and cannot be used as a identifier name"));

	private static readonly IEnumerable<string> KeywordsList =
	[
		"has", "mutable", "constant", "if", "else", "for", "with", "return"
	];

	[TestCaseSource(nameof(KeywordsList))]
	public void CannotUseKeywordsAsVariableName(string name) =>
		Assert.That(
			() => new Type(package,
					new TypeLines(name + nameof(CannotUseKeywordsAsVariableName), "has number", "Run",
						$"\tconstant {name} = 5")).ParseMembersAndMethods(parser).Methods[0].
				GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<CannotUseKeywordsAsName>());

	[TestCaseSource(nameof(KeywordsList))]
	public void CannotUseKeywordsAsMethodParameterName(string name) =>
		Assert.That(() => new Type(package,
			new TypeLines(name + nameof(CannotUseKeywordsAsMethodParameterName),
				"has Number",
				$"Run(mutable {name} Number)",
				"\t5")).ParseMembersAndMethods(parser), Throws.InstanceOf<ParsingFailed>().With.InnerException.InstanceOf<CannotUseKeywordsAsName>().With.Message.Contains($"{name} is a keyword and cannot be used as a identifier name"));

	[TestCaseSource(nameof(KeywordsList))]
	public void CannotUseKeywordsAsMemberOrMethodExpressions(string name) =>
		Assert.That(
			() => new Type(package,
				new TypeLines(
					name.MakeFirstLetterUppercase() + nameof(CannotUseKeywordsAsMethodParameterName),
					"has input = if", "Run(number)", "\t5")).ParseMembersAndMethods(parser),
			Throws.InstanceOf<MethodExpressionParser.KeywordNotAllowedAsMemberOrMethod>());
}