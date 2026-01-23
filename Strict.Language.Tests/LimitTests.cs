using System.Text;

namespace Strict.Language.Tests;

public sealed class LimitTests
{
	[SetUp]
	public void CreatePackage() => parser = new MethodExpressionParser();

	private MethodExpressionParser parser = null!;

	[Test]
	public void MethodLengthMustNotExceedTwelve() =>
		Assert.That(
			() =>
			{
				using var _ = CreateType(nameof(MethodLengthMustNotExceedTwelve),
					CreateProgramWithDuplicateLines(["has logger", "Run(first Number, second Number)"], 12,
						"\tlogger.Log(5)")).ParseMembersAndMethods(parser);
			},
			Throws.InstanceOf<Method.MethodLengthMustNotExceedTwelve>().With.Message.
				Contains($"Method Run has 13 lines but limit is {Limit.MethodLength}"));

	private Type CreateType(string name, string[] lines) =>
		new Type(TestPackage.Instance, new TypeLines(name, lines)).ParseMembersAndMethods(parser);

	private static string[] CreateProgramWithDuplicateLines(string[] defaultLines, int count,
		params string[] linesToDuplicate)
	{
		var program = new string[defaultLines.Length + count * linesToDuplicate.Length];
		defaultLines.CopyTo(program, 0);
		CreateDuplicateLines(count, linesToDuplicate).CopyTo(program, defaultLines.Length);
		return program;
	}

	private static string[] CreateDuplicateLines(int count, params string[] lines)
	{
		var outputLines = new string[count * lines.Length];
		for (var index = 0; index < count; index++)
			lines.CopyTo(outputLines, index * lines.Length);
		return outputLines;
	}

	[Test]
	public void MethodParameterCountMustNotExceedLimit() =>
		Assert.That(() => CreateType(nameof(MethodParameterCountMustNotExceedLimit), [
				"has logger",
				"Run(first Number, second Number, third Number, fourth Number, fifth Number)",
				"\tlogger.Log(5)"
			]),
			Throws.InstanceOf<Method.MethodParameterCountMustNotExceedLimit>().With.Message.
				Contains($"Method Run has parameters count 5 but limit is {Limit.ParameterCount}"));

	[Test]
	public void MethodCountMustNotExceedFifteen() =>
		Assert.That(() =>
			{
				using var _ = CreateType(nameof(MethodCountMustNotExceedFifteen), [
				// @formatter:off
					"has logger",
					"RunFirst", "\tlogger.Log(1)",
					"RunSecond", "\tlogger.Log(2)",
					"RunThird", "\tlogger.Log(3)",
					"RunForth", "\tlogger.Log(4)",
					"RunFifth", "\tlogger.Log(5)",
					"RunSixth", "\tlogger.Log(6)",
					"RunSeventh", "\tlogger.Log(7)",
					"RunEight", "\tlogger.Log(8)",
					"RunNine", "\tlogger.Log(9)",
					"RunTen", "\tlogger.Log(10)",
					"RunEleven", "\tlogger.Log(11)",
					"RunTwelve", "\tlogger.Log(12)",
					"RunThirteen", "\tlogger.Log(13)",
					"RunFourteen", "\tlogger.Log(14)",
					"RunFifteen", "\tlogger.Log(15)",
					"RunSixteen", "\tlogger.Log(16)"
					// @formatter:on
				]);
			},
			Throws.InstanceOf<Type.MethodCountMustNotExceedLimit>().With.Message.Contains(
				$"Type MethodCountMustNotExceedFifteen has method count 16 but limit is {
					Limit.MethodCount
				}"));

	[Test]
	public void LinesCountMustNotExceedTwoHundredFiftySix() =>
		Assert.That(
			() =>
			{
				using var _ = CreateType(nameof(LinesCountMustNotExceedTwoHundredFiftySix),
					CreateDuplicateLines(257, "has logger").ToArray()).ParseMembersAndMethods(parser);
			},
			Throws.InstanceOf<Type.LinesCountMustNotExceedLimit>().With.Message.Contains(
				$"Type LinesCountMustNotExceedTwoHundredFiftySix has lines count 257 but limit is {
					Limit.LineCount
				}"));

	[Test]
	public void NestingMoreThanFiveLevelsIsNotAllowed() =>
		Assert.That(() =>
			{
				using var _ = CreateType(nameof(NestingMoreThanFiveLevelsIsNotAllowed), [
				// @formatter:off
				"has logger",
				"Run",
				"	if 5 is 5",
				"		if 6 is 6",
				"			if 7 is 7",
				"				if 8 is 8",
				"					if 9 is 9",
				"						logger.Log(5)" // @formatter:on
				]).ParseMembersAndMethods(parser);
			},
			Throws.InstanceOf<TypeParser.NestingMoreThanFiveLevelsIsNotAllowed>().With.Message.Contains(
				$"Type NestingMoreThanFiveLevelsIsNotAllowed has more than {
					Limit.NestingLevel
				} levels of nesting in line: 8"));

	[Test]
	public void CharacterCountMustBeWithinLimit() =>
		Assert.That(() =>
			{
				using var _ = CreateType(nameof(CharacterCountMustBeWithinLimit), [
					"has bonus Number", "has price Number",
					"CalculateCompleteLevelCount(numberOfCans Number, levelCount Number) Number",
					"	constant remainingCans = numberOfCans - (levelCount * levelCount)remainingCans < " +
					"((levelCount + 1) * (levelCount + 1)) ? levelCount else CalculateCompleteLevelCount(" +
					"remainingCans, levelCount + 1)"
				]).ParseMembersAndMethods(parser);
			},
			Throws.InstanceOf<TypeParser.CharacterCountMustBeWithinLimit>().With.Message.Contains(
				"Type " + nameof(CharacterCountMustBeWithinLimit) +
				" has character count 196 in line: 4 but limit is " + Limit.CharacterCount));

	[Test]
	public void MemberCountShouldNotExceedLimit() =>
		Assert.That(
			() =>
			{
				using var _ = CreateType(nameof(MemberCountShouldNotExceedLimit),
					CreateRandomMemberLines(Limit.MemberCountForEnums + 1)).ParseMembersAndMethods(parser);
			},
			Throws.InstanceOf<Type.MemberCountShouldNotExceedLimit>().With.Message.Contains(
				nameof(Type.MemberCountShouldNotExceedLimit) + " type has " +
				(Limit.MemberCountForEnums + 1) + " members, max: " + Limit.MemberCountForEnums));

	private static string[] CreateRandomMemberLines(int count)
	{
		var lines = new string[count];
		var random = new Random();
		for (var index = 0; index < count; index++)
			lines[index] = "constant " + GetRandomMemberName(random, 6);
		return lines;
	}

	private static string GetRandomMemberName(Random random, int size)
	{
		var result = new StringBuilder();
		for (var i = 0; i < size; i++)
			result.Append((char)random.Next('a', 'a' + 26));
		return result.ToString();
	}

	[TestCase("Member", "memberNameSomethingWithLengthGreaterThanFiftyIsNotAllowed",
		"has memberNameSomethingWithLengthGreaterThanFiftyIsNotAllowed Number", "Run", "	5")]
	[TestCase("MemberMinimum", "m", "has m Number", "Run", "	5")]
	[TestCase("Parameter", "parameterNameGreaterThanFiftyExceedsLimitNotAllowed", "has number",
		"Run(parameterNameGreaterThanFiftyExceedsLimitNotAllowed Number)", "	5")]
	[TestCase("ParameterMinimum", "p", "has number", "Run(p Number)", "	5")]
	public void
		NameShouldBeWithinTheLimit(string testName, string memberOrParameterName,
			params string[] code) =>
		Assert.That(
			() =>
			{
				using var _ = CreateType(testName + nameof(NameShouldBeWithinTheLimit), code).
					ParseMembersAndMethods(parser);
			},
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<NamedType.NameLengthIsNotWithinTheAllowedLimit>().With.Message.Contains(
					$"Name {memberOrParameterName} " + $"length is {
						memberOrParameterName.Length
					} but allowed limit is between 2 and 50"));

	[TestCase("TypeNameWithLengthGreaterThanFiftyIsNotAllowedUseWithinLimit")]
	[TestCase("T")]
	public void TypeNameShouldNotExceedTheLimit(string typeName) =>
		Assert.That(
			() =>
			{
				using var _ = CreateType(typeName, ["has number", "Run", "\t5"]).ParseMembersAndMethods(parser);
			},
			Throws.InstanceOf<NamedType.NameLengthIsNotWithinTheAllowedLimit>().With.Message.Contains(
				$"Name {typeName} length is {typeName.Length} but allowed limit is between 2 and 50"));

	[Test]
	public void VariableNameShouldNotExceedTheLimit() =>
		Assert.That(
			() =>
			{
				using var type = CreateType(nameof(VariableNameShouldNotExceedTheLimit), [
					"has number", "Run",
					"\tconstant variablesNameWithLengthGreaterThanFiftyAreNotAllowed = 5"
				]);
				type.Methods[0].GetBodyAndParseIfNeeded();
			},
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<NamedType.NameLengthIsNotWithinTheAllowedLimit>().With.Message.
				Contains("constant variablesNameWithLengthGreaterThanFiftyAreNotAllowed"));

	[Test]
	public void VariableNameShouldNotBeBelowTheLimit() =>
		Assert.That(
			() =>
			{
				using var type = CreateType(nameof(VariableNameShouldNotBeBelowTheLimit),
					["has number", "Run", "\tconstant v = 5"]);
				return type.Methods[0].GetBodyAndParseIfNeeded();
			},
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<NamedType.NameLengthIsNotWithinTheAllowedLimit>().With.Message.
				Contains("constant v"));

	[TestCase(nameof(PackageNameShouldBeWithinTheLimit) + "LimitShouldBeWithinFifty")]
	[TestCase("P")]
	public void PackageNameShouldBeWithinTheLimit(string name) =>
		Assert.That(() => new Package(name),
			Throws.InstanceOf<NamedType.NameLengthIsNotWithinTheAllowedLimit>().With.Message.
				Contains($"Name {name} length is {name.Length} but allowed limit is between 2 and 50"));

	[TestCase("MethodNameWithLengthGreaterThanFiftyExceedsLimitAndIsNotAllowed")]
	[TestCase("M")]
	public void MethodNameShouldNotExceedTheLimit(string methodName) =>
		Assert.That(() =>
			{
				using var _ = CreateType(nameof(MethodNameShouldNotExceedTheLimit) + methodName.Last(), [
					"has number", methodName, "	constant number = 5"
				]).ParseMembersAndMethods(parser);
			},
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<NamedType.NameLengthIsNotWithinTheAllowedLimit>().With.Message.Contains(
					$"Name {
						methodName
					} length is {
						methodName.Length
					} but allowed limit is between 2 and 50"));
}