using System.Text;
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
				CreateProgramWithDuplicateLines(new[] { "has log", "Run(first Number, second Number)" },
					12, "\tlog.Write(5)")).ParseMembersAndMethods(new MethodExpressionParser()),
			Throws.InstanceOf<Method.MethodLengthMustNotExceedTwelve>().With.Message.
				Contains($"Method Run has 13 lines but limit is {Limit.MethodLength}"));

	private Type CreateType(string name, string[] lines) =>
		new Type(package, new TypeLines(name, lines)).ParseMembersAndMethods(
			new MethodExpressionParser());

	private static string[] CreateProgramWithDuplicateLines(IEnumerable<string> defaultLines,
		int count, params string[] linesToDuplicate)
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
			() => CreateType(nameof(MethodParameterCountMustNotExceedThree),
				new[]
				{
					"has log", "Run(first Number, second Number, third Number, fourth Number)",
					"\tlog.Write(5)"
				}).ParseMembersAndMethods(new MethodExpressionParser()),
			Throws.InstanceOf<Method.MethodParameterCountMustNotExceedThree>().With.Message.
				Contains($"Method Run has parameters count 4 but limit is {Limit.ParameterCount}"));

	[Test]
	public void MethodCountMustNotExceedFifteen() =>
		Assert.That(
			() => CreateType(nameof(MethodCountMustNotExceedFifteen),
					CreateProgramWithDuplicateLines(new[] { "has log" }, 16,
						"Run(first Number, second Number)", "\tfirst")).
				ParseMembersAndMethods(new MethodExpressionParser()),
			Throws.InstanceOf<Type.MethodCountMustNotExceedLimit>().With.Message.Contains(
				$"Type MethodCountMustNotExceedFifteen has method count 16 but limit is {
					Limit.MethodCount
				}"));

	[Test]
	public void LinesCountMustNotExceedTwoHundredFiftySix() =>
		Assert.That(
			() => CreateType(nameof(LinesCountMustNotExceedTwoHundredFiftySix),
					CreateDuplicateLines(257, "has log").ToArray()).
				ParseMembersAndMethods(new MethodExpressionParser()),
			Throws.InstanceOf<Type.LinesCountMustNotExceedLimit>().With.Message.Contains(
				$"Type LinesCountMustNotExceedTwoHundredFiftySix has lines count 257 but limit is {
					Limit.LineCount
				}"));

	[Test]
	public void NestingMoreThanFiveLevelsIsNotAllowed() =>
		Assert.That(() => CreateType(nameof(NestingMoreThanFiveLevelsIsNotAllowed), new[]
			{
				// @formatter:off
				"has log",
				"Run",
				"	if 5 is 5",
				"		if 6 is 6",
				"			if 7 is 7",
				"				if 8 is 8",
				"					if 9 is 9",
				"						log.Write(5)" // @formatter:on
			}).ParseMembersAndMethods(new MethodExpressionParser()),
			Throws.InstanceOf<TypeParser.NestingMoreThanFiveLevelsIsNotAllowed>().With.Message.Contains(
				$"Type NestingMoreThanFiveLevelsIsNotAllowed has more than {
					Limit.NestingLevel
				} levels of nesting in line: 8"));

	[Test]
	public void CharacterCountMustBeWithinLimit() =>
		Assert.That(
			() => CreateType(nameof(CharacterCountMustBeWithinLimit),
				new[]
				{
					"has bonus Number", "has price Number",
					"CalculateCompleteLevelCount(numberOfCans Number, levelCount Number) Number",
					"	constant remainingCans = numberOfCans - (levelCount * levelCount)remainingCans < ((levelCount + 1) * (levelCount + 1)) ? levelCount else CalculateCompleteLevelCount(remainingCans, levelCount + 1)"
				}).ParseMembersAndMethods(new MethodExpressionParser()),
			Throws.InstanceOf<TypeParser.CharacterCountMustBeWithinLimit>().With.Message.Contains("Type " +
				nameof(CharacterCountMustBeWithinLimit) +
				" has character count 196 in line: 4 but limit is " + Limit.CharacterCount));

	[Test]
	public void MemberCountShouldNotExceedLimit() =>
		Assert.That(
			() => CreateType(nameof(MemberCountShouldNotExceedLimit),
					CreateRandomMemberLines(Limit.MemberCountForEnums + 1)).
				ParseMembersAndMethods(new MethodExpressionParser()),
			Throws.InstanceOf<Type.MemberCountShouldNotExceedLimit>().With.Message.Contains(
				nameof(Type.MemberCountShouldNotExceedLimit) + " type has " +
				(Limit.MemberCountForEnums + 1) + " members, max: " + Limit.MemberCountForEnums));

	private static string[] CreateRandomMemberLines(int count)
	{
		var lines = new string[count];
		var random = new Random();
		for (var index = 0; index < count; index++)
			lines[index] = "has " + GetRandomMemberName(random, 6) + " = 1";
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
			() => CreateType(testName + nameof(NameShouldBeWithinTheLimit), code).
				ParseMembersAndMethods(new MethodExpressionParser()),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<NamedType.NameLengthIsNotWithinTheAllowedLimit>().With.Message.Contains($"Name {memberOrParameterName} " +
					$"length is {memberOrParameterName.Length} but allowed limit is between 2 and 50"));

	[TestCase("TypeNameWithLengthGreaterThanFiftyIsNotAllowedUseWithinLimit")]
	[TestCase("T")]
	public void TypeNameShouldNotExceedTheLimit(string typeName) =>
		Assert.That(
			() => CreateType(typeName, new[] { "has number", "Run", "\t5" }).
				ParseMembersAndMethods(new MethodExpressionParser()),
			Throws.InstanceOf<NamedType.NameLengthIsNotWithinTheAllowedLimit>().With.Message.Contains(
				$"Name {typeName} length is {typeName.Length} but allowed limit is between 2 and 50"));

	[TestCase("variablesNameWithLengthGreaterThanFiftyAreNotAllowed")]
	[TestCase("v")]
	public void VariableNameShouldNotExceedTheLimit(string variableName) =>
		Assert.That(
			() => CreateType(nameof(VariableNameShouldNotExceedTheLimit),
					new[]
					{
						"has number", "Run",
						$"	constant {variableName} = 5"
					}).ParseMembersAndMethods(new MethodExpressionParser()).Methods[0].
				GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.InstanceOf<NamedType.NameLengthIsNotWithinTheAllowedLimit>().With.Message.Contains(
				$"constant {variableName}"));

	[TestCase(nameof(PackageNameShouldBeWithinTheLimit) + "LimitShouldBeWithinFifty")]
	[TestCase("P")]
	public void PackageNameShouldBeWithinTheLimit(string name) =>
		Assert.That(() => new Package(name),
			Throws.InstanceOf<NamedType.NameLengthIsNotWithinTheAllowedLimit>().With.Message.
				Contains($"Name {name} length is {name.Length} but allowed limit is between 2 and 50"));

	[TestCase("MethodNameWithLengthGreaterThanFiftyExceedsLimitAndIsNotAllowed")]
	[TestCase("M")]
	public void MethodNameShouldNotExceedTheLimit(string methodName) =>
		Assert.That(
			() => CreateType(nameof(MethodNameShouldNotExceedTheLimit),
					new[] { "has number", methodName, "	constant number = 5" }).
				ParseMembersAndMethods(new MethodExpressionParser()),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<NamedType.NameLengthIsNotWithinTheAllowedLimit>().With.Message.Contains(
					$"Name {
						methodName
					} length is {
						methodName.Length
					} but allowed limit is between 2 and 50"));
}