using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Running;
using Strict.Expressions;
using Strict.HighLevelRuntime;
using Strict.Language;
using Strict.Language.Tests;
using Type = Strict.Language.Type;

namespace Strict.TestRunner.Tests;

[MemoryDiagnoser]
[SimpleJob(RunStrategy.Throughput, warmupCount: 1, iterationCount: 10)]
public sealed class TestExecutorTests
{
	private readonly TestExecutor executor = new();

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
					"\tvalue is in Range(0, 10) then Character(Character.zeroCharacter + value) else canOnlyConvertSingleDigit(value)")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(type.Methods[0].GetBodyAndParseIfNeeded().ToString(),
			Is.EqualTo(new[]
			{
				"5 to Character is \"5\"",
				"constant canOnlyConvertSingleDigit = Error",
				"13 to Character is canOnlyConvertSingleDigit",
				"value is in Range(0, 10) then Character(Character.zeroCharacter + value) else canOnlyConvertSingleDigit(value)"
				// @formatter:on
			}.ToWordList(Environment.NewLine)));
		executor.RunAllTestsInType(type);
	}

	[Test]
	public void RunRangeSum()
	{
		using var type = new Type(TestPackage.Instance,
				new TypeLines(nameof(RunNumberToCharacterBody), "has number",
					// @formatter:off
					"SumRange(range) Number",
					"\tSumRange(Range(2, 5)) is 2 + 3 + 4",
					"\tSumRange(Range(42, 45)) is 42 + 43 + 44",
					"\tfor range",
					"\t\tvalue")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(type.Methods[0].GetBodyAndParseIfNeeded().ToString(),
			Is.EqualTo(new[]
			{
				"SumRange(Range(2, 5)) is 2 + 3 + 4",
				"SumRange(Range(42, 45)) is 42 + 43 + 44",
				"for range",
				"\tvalue"
				// @formatter:on
			}.ToWordList(Environment.NewLine)));
		executor.RunAllTestsInType(type);
	}

	[Test]
	public void RunListLength()
	{
		using var type = new Type(TestPackage.Instance,
				new TypeLines(nameof(RunNumberToCharacterBody), "has numbers",
					// @formatter:off
					"Length Number",
					"\tfor numbers",
					"\t\t1")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(type.Methods[0].GetBodyAndParseIfNeeded().ToString(),
			Is.EqualTo(new[]
			{
				"for numbers",
				"\t1"
				// @formatter:on
			}.ToWordList(Environment.NewLine)));
		executor.RunAllTestsInType(type);
	}

	[Test]
	public void RunAddLists()
	{
		using var type = new Type(TestPackage.Instance,
				new TypeLines(nameof(RunNumberToCharacterBody), "has number",
					"Run Number",
					"\t(1, 2, 3) + (4, 5) is (1, 2, 3, 4, 5)",
					"\t(\"Hello\", \"World\") + (1, 2) is (\"Hello\", \"World\", \"1\", \"2\")",
					"\tnumber")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(type.Methods[0].GetBodyAndParseIfNeeded().ToString(),
			Is.EqualTo("(1, 2, 3) + (4, 5) is (1, 2, 3, 4, 5)" + Environment.NewLine +
				"(\"Hello\", \"World\") + (1, 2) is (\"Hello\", \"World\", \"1\", \"2\")" +
				Environment.NewLine + "number"));
		executor.RunAllTestsInType(type);
	}

	[Test]
	public void RunMutableListCompare()
	{
		using var type = new Type(TestPackage.Instance,
			new TypeLines(nameof(RunNumberToCharacterBody), "has number", "Run",
				"\t(1, 2).Add(3) is (1, 2, 3)")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(type.Methods[0].GetBodyAndParseIfNeeded().ToString(),
			Is.EqualTo("(1, 2).Add(3) is (1, 2, 3)"));
		executor.RunAllTestsInType(type);
	}

	[Test]
	public void RunTextCompare()
	{
		using var type = new Type(TestPackage.Instance,
			new TypeLines(nameof(RunNumberToCharacterBody), "has number", "Run",
				"\t\"Hey\" is \"Hey\"",
				"\t\"Hi\" is not \"Hey\"",
				"\tnumber")).ParseMembersAndMethods(new MethodExpressionParser());
		executor.RunAllTestsInType(type);
	}

	[Test]
	public void CompareMutableList()
	{
		using var type = new Type(TestPackage.Instance,
				new TypeLines(nameof(RunNumberToCharacterBody), "has number", "Run",
					"\t(1, 2).Add(3) is (1, 2, 3)", "\tnumber")).
			ParseMembersAndMethods(new MethodExpressionParser());
		executor.RunAllTestsInType(type);
	}

	[Test]
	[Benchmark]
	public void RunAllTestsInPackage() => executor.RunAllTestsInPackage(TestPackage.Instance);

	//ncrunch: no coverage start
	[Test]
	[Category("Manual")]
	public void BenchmarkCompare() => BenchmarkRunner.Run<TestExecutorTests>();
}