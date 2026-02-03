using Strict.Expressions;
using Strict.HighLevelRuntime;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.TestRunner.Tests;

public sealed class TestExecutorTests
{
	[SetUp]
	public void Setup() => executor = new TestExecutor(TestPackage.Instance);

	private TestExecutor executor = null!;

	[Test]
	public void RunMethod()
	{
		using var type = new Type(TestPackage.Instance,
			new TypeLines(nameof(RunMethod),
				"has number",
				"Run Number",
				"	5 is 5",
				"	10")).ParseMembersAndMethods(new MethodExpressionParser());
		executor.RunMethod(type.Methods.First(m => m.Name == "Run"));
	}

	[Test]
	public void RunMethodWithFailingTest()
	{
		using var type = new Type(TestPackage.Instance,
			new TypeLines(nameof(RunMethodWithFailingTest),
				"has number",
				"Run Number",
				"	5 is 6",
				"	10")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(() => executor.RunMethod(type.Methods.First(m => m.Name == "Run")),
			Throws.InstanceOf<ExecutionFailed>().With.InnerException.With.Message.
				StartsWith("\"Run\" method failed: 5 is 6, result: False"));
	}

	[Test]
	public void RunAllTestsInType()
	{
		using var type = new Type(TestPackage.Instance,
			new TypeLines(nameof(RunAllTestsInType),
				"has number",
				"Run Number",
				"	5 is 5",
				"	10",
				"Other Number",
				"	2 is 2",
				"	5")).ParseMembersAndMethods(new MethodExpressionParser());
		executor.RunAllTestsInType(type);
	}

	[Test]
	public void RunAllTestsInTypeWithFailure()
	{
		using var type = new Type(TestPackage.Instance,
			new TypeLines(nameof(RunAllTestsInTypeWithFailure),
				"has number",
				"Run Number",
				"	5 is 5",
				"	10",
				"Other Number",
				"	2 is 3",
				"	5")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(() => executor.RunAllTestsInType(type),
			Throws.InstanceOf<ExecutionFailed>().With.InnerException.With.Message.
				Contains("\"Other\" method failed: 2 is 3, result: False"));
	}

	[Test]
	public void RunListCreation()
	{
		using var type = new Type(TestPackage.Instance,
				new TypeLines(nameof(RunListCreation), "has number", "Run",
					"	constant numberList = List(5)",
					"	numberList(0) is number")).
			ParseMembersAndMethods(new MethodExpressionParser());
		executor.RunAllTestsInType(type);
	}

	[Test]
	public void RunErrorComparison()
	{
		using var type = new Type(TestPackage.Instance,
				new TypeLines(nameof(RunErrorComparison), "has number", "Run",
					"	constant canOnlyConvertSingleDigit = Error",
					"	13 to Character is canOnlyConvertSingleDigit")).
			ParseMembersAndMethods(new MethodExpressionParser());
		executor.RunAllTestsInType(type);
	}

	[Test]
	public void RunRangeReverseComparison()
	{
		using var type = new Type(TestPackage.Instance,
				new TypeLines(nameof(RunRangeReverseComparison), "has number", "Run",
					"	Range(-5, -10).Reverse is Range(-9, -4)")).
			ParseMembersAndMethods(new MethodExpressionParser());
		executor.RunAllTestsInType(type);
	}

	[Test]
	public void RunNumberToCharacterBody()
	{
		using var type = new Type(TestPackage.Instance,
				new TypeLines(nameof(RunNumberToCharacterBody), "has number",
					// @formatter:off
					"to Character",
					"\t5 to Character is \"5\"",
					"\tconstant canOnlyConvertSingleDigit = Error",
					"\t13 to Character is canOnlyConvertSingleDigit",
					"\tvalue is in Range(0, 10) ? Character(Character.zeroCharacter + value) else canOnlyConvertSingleDigit(value)")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(type.Methods[0].GetBodyAndParseIfNeeded().ToString(),
			Is.EqualTo(new[]
			{
				"5 to Character is \"5\"",
				"constant canOnlyConvertSingleDigit = Error",
				"13 to Character is canOnlyConvertSingleDigit",
				"value is in Range(0, 10) ? Character(Character.zeroCharacter + value) else canOnlyConvertSingleDigit(value)"
				// @formatter:on
			}.ToWordList(Environment.NewLine)));
		executor.RunAllTestsInType(type);
	}

	//TODO: [Test]
	//public void RunAllTestsInPackage() => executor.RunAllTestsInPackage(TestPackage.Instance);
}