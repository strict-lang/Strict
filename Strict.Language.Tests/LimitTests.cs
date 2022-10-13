using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using Strict.Language.Expressions;

namespace Strict.Language.Tests;

public sealed class LimitTests
{
	[SetUp]
	public void CreatePackage() => package = new TestPackage();

	private Package package = null!;

	[Test]
	public void MethodLengthMustNotExceedTwelve() =>
		Assert.That(
			() => CreateType(nameof(MethodLengthMustNotExceedTwelve),
					CreateProgramWithDuplicateLines(
						new[] { "has log", "Run(first Number, second Number)" }, 12, "\tlog.Write(5)")).
				ParseMembersAndMethods(new MethodExpressionParser()),
			Throws.InstanceOf<Method.MethodLengthMustNotExceedTwelve>().With.Message.
				Contains("Method Run has 13 lines but limit is 12"));

	private Type CreateType(string name, params string[] lines) =>
		new Type(package, new TypeLines(name, lines)).ParseMembersAndMethods(new MethodExpressionParser());

	private static string[] CreateProgramWithDuplicateLines(IEnumerable<string> defaultLines, int count, params string[] linesToDuplicate)
	{
		var program = new List<string>();
		program.AddRange(defaultLines);
		program.AddRange(CreateDuplicateLines(count, linesToDuplicate));
		return program.ToArray();
	}

	private static IReadOnlyList<string> CreateDuplicateLines(int count, params string[] lines)
	{
		var outputLines = new List<string>();
		for (var index = 0; index < count; index++)
			outputLines.AddRange(lines);
		return outputLines;
	}

	[Test]
	public void MethodParameterCountMustNotExceedThree() =>
		Assert.That(
			() => CreateType(nameof(MethodParameterCountMustNotExceedThree), "has log",
					"Run(first Number, second Number, third Number, fourth Number)", "\tlog.Write(5)").
				ParseMembersAndMethods(new MethodExpressionParser()),
			Throws.InstanceOf<Method.MethodParameterCountMustNotExceedThree>().With.Message.
				Contains("Method Run has parameters count 4 but limit is 3"));

	[Test]
	public void MembersCountMustNotExceedTen() =>
		Assert.That(
			() => CreateType(nameof(MembersCountMustNotExceedTen), CreateDuplicateLines(11, "has log").ToArray()).
				ParseMembersAndMethods(new MethodExpressionParser()),
			Throws.InstanceOf<Type.MembersCountMustNotExceedTen>().With.Message.
				Contains("Type MembersCountMustNotExceedTen has member count 11 but limit is 10"));

	[Test]
	public void MethodCountMustNotExceedFifteen() =>
		Assert.That(
			() => CreateType(nameof(MethodCountMustNotExceedFifteen), CreateProgramWithDuplicateLines(
					new[] { "has log" }, 16, "Run(first Number, second Number)", "\tfirst")).
				ParseMembersAndMethods(new MethodExpressionParser()),
			Throws.InstanceOf<Type.MethodCountMustNotExceedFifteen>().With.Message.
				Contains("Type MethodCountMustNotExceedFifteen has method count 16 but limit is 15"));

	[Test]
	public void LinesCountMustNotExceedTwoHundredFiftySix() =>
		Assert.That(
			() => CreateType(nameof(LinesCountMustNotExceedTwoHundredFiftySix),
					CreateDuplicateLines(257, "has log").ToArray()).
				ParseMembersAndMethods(new MethodExpressionParser()),
			Throws.InstanceOf<Type.LinesCountMustNotExceedTwoHundredFiftySix>().With.Message.Contains(
				"Type LinesCountMustNotExceedTwoHundredFiftySix has lines count 257 but limit is 256"));

	[Test]
	public void NestingMoreThanFiveLevelsIsNotAllowed() =>
		Assert.That(() => CreateType(nameof(NestingMoreThanFiveLevelsIsNotAllowed),
				// @formatter:off
				"has log",
				"Run",
				"	if 5 is 5",
				"		if 6 is 6",
				"			if 7 is 7",
				"				if 8 is 8",
				"					if 9 is 9",
				"						log.Write(5)")
				// @formatter:on
				.ParseMembersAndMethods(new MethodExpressionParser()),
			Throws.InstanceOf<Type.NestingMoreThanFiveLevelsIsNotAllowed>().With.Message.Contains(
				"Type NestingMoreThanFiveLevelsIsNotAllowed has more than 5 levels of nesting in line: 8"));

	[Test]
	public void CharacterCountMustBeWithinOneHundredTwenty() =>
		Assert.That(() => CreateType(nameof(CharacterCountMustBeWithinOneHundredTwenty),
				"has bonus Number",
				"has price Number",
				"CalculateCompleteLevelCount(numberOfCans Number, levelCount Number) Number",
				"	let remainingCans = numberOfCans - (levelCount * levelCount)remainingCans < ((levelCount + 1) * (levelCount + 1)) ? levelCount else CalculateCompleteLevelCount(remainingCans, levelCount + 1)").ParseMembersAndMethods(new MethodExpressionParser()),
			Throws.InstanceOf<Type.CharacterCountMustBeWithinOneHundredTwenty>().With.Message.Contains(
				"Type CharacterCountMustBeWithinOneHundredTwenty has character count 191 in line: 4 but limit is 256"));
}