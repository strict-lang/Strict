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
public class TestInterpreterTests
{
	public readonly TestInterpreter interpreter = new(TestPackage.Instance);

	[Test]
	public void RunMethod()
	{
		using var type = new Type(TestPackage.Instance,
			new TypeLines(nameof(RunMethod),
				"has number",
				"Run Number",
				"	5 is 5",
				"	10")).ParseMembersAndMethods(new MethodExpressionParser());
		interpreter.RunMethod(type.Methods.First(m => m.Name == "Run"));
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
		Assert.That(() => interpreter.RunMethod(type.Methods.First(m => m.Name == "Run")),
			Throws.InstanceOf<InterpreterExecutionFailed>().With.InnerException.With.Message.
				Contains("\"Run\" method failed: 5 is 6, result: Boolean: false"));
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
		interpreter.RunAllTestsInType(type);
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
		Assert.That(() => interpreter.RunAllTestsInType(type),
			Throws.InstanceOf<InterpreterExecutionFailed>().With.InnerException.With.Message.
				Contains("\"Other\" method failed: 2 is 3, result: Boolean: false"));
	}

	[Test]
	public void RunListCreation()
	{
		using var type = new Type(TestPackage.Instance,
				new TypeLines(nameof(RunListCreation), "has number", "Run",
					"	constant numberList = List(5)",
					"	numberList(0) is number")).
			ParseMembersAndMethods(new MethodExpressionParser());
		interpreter.RunAllTestsInType(type);
	}

	[Test]
	public void RunErrorComparison()
	{
		using var type = new Type(TestPackage.Instance,
				new TypeLines(nameof(RunErrorComparison), "has number", "Run",
					"	constant canOnlyConvertSingleDigit = Error",
					"	13 to Character is canOnlyConvertSingleDigit")).
			ParseMembersAndMethods(new MethodExpressionParser());
		interpreter.RunAllTestsInType(type);
	}

	[Test]
	public void RunRangeReverseComparison()
	{
		using var type = new Type(TestPackage.Instance,
				new TypeLines(nameof(RunRangeReverseComparison), "has number", "Run",
					"	Range(-5, -10).Reverse is Range(-9, -4)")).
			ParseMembersAndMethods(new MethodExpressionParser());
		interpreter.RunAllTestsInType(type);
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
			}.ToLines()));
		interpreter.RunAllTestsInType(type);
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
					"\t\tindex")).
			ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(type.Methods[0].GetBodyAndParseIfNeeded().ToString(),
			Is.EqualTo(new[]
			{
				"SumRange(Range(2, 5)) is 2 + 3 + 4",
				"SumRange(Range(42, 45)) is 42 + 43 + 44",
				"for range",
				"\tindex"
				// @formatter:on
			}.ToLines()));
		interpreter.RunAllTestsInType(type);
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
			}.ToLines()));
		interpreter.RunAllTestsInType(type);
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
		interpreter.RunAllTestsInType(type);
	}

	[Test]
	public void RunMutableListCompare()
	{
		using var type = new Type(TestPackage.Instance,
			new TypeLines(nameof(RunNumberToCharacterBody), "has number", "Run",
				"\t(1, 2).Add(3) is (1, 2, 3)")).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(type.Methods[0].GetBodyAndParseIfNeeded().ToString(),
			Is.EqualTo("(1, 2).Add(3) is (1, 2, 3)"));
		interpreter.RunAllTestsInType(type);
	}

	[Test]
	public void RunTextCompare()
	{
		using var type = new Type(TestPackage.Instance,
			new TypeLines(nameof(RunNumberToCharacterBody), "has number", "Run",
				"\t\"Hey\" is \"Hey\"",
				"\t\"Hi\" is not \"Hey\"",
				"\tnumber")).ParseMembersAndMethods(new MethodExpressionParser());
		interpreter.RunAllTestsInType(type);
	}

	[Test]
	public void CompareMutableList()
	{
		using var type = new Type(TestPackage.Instance,
				new TypeLines(nameof(RunNumberToCharacterBody), "has number", "Run",
					"\t(1, 2).Add(3) is (1, 2, 3)", "\tnumber")).
			ParseMembersAndMethods(new MethodExpressionParser());
		interpreter.RunAllTestsInType(type);
	}

	[Test]
	public void RunAllTestsInTypeWithLoggerAutoInjected()
	{
		using var type = new Type(TestPackage.Instance,
			new TypeLines(nameof(RunAllTestsInTypeWithLoggerAutoInjected),
				"has first Number",
				"has second Number",
				"has logger",
				"Add Number",
				"\tRunAllTestsInTypeWithLoggerAutoInjected(2, 3).Add is 5",
				"\tfirst + second")).ParseMembersAndMethods(new MethodExpressionParser());
		interpreter.RunAllTestsInType(type);
	}

	[Test]
	public void RunDictionaryTestsTwice()
	{
		using var type = TestPackage.Instance.GetType(Type.Dictionary);
		interpreter.RunAllTestsInType(type);
		interpreter.RunAllTestsInType(type);
	}

	[Test]
	public void RunColorToColorValueTest()
	{
		using var colorValueType = new Type(TestPackage.Instance,
			new TypeLines("ColorValue", "has Red Number", "has Green Number", "has Blue Number",
				"has Alpha = 1")).ParseMembersAndMethods(new MethodExpressionParser());
		using var colorType = new Type(TestPackage.Instance,
				new TypeLines("Color", "has Red Byte", "has Green Byte", "has Blue Byte",
					"has Alpha Byte = 255", "to ColorValue",
					"\tColor(255, 0, 0) to ColorValue is ColorValue(1, 0, 0)",
					"\tColorValue(Red / 255, Green / 255, Blue / 255, Alpha / 255)")).
			ParseMembersAndMethods(new MethodExpressionParser());
		var numberType = colorType.GetType(Type.Number);
		interpreter.RunAllTestsInType(colorType);
		var result = interpreter.Execute(colorType.GetMethod("to", []),
			new ValueInstance(colorType,
			[
				new ValueInstance(numberType, 0), new ValueInstance(numberType, 127),
				new ValueInstance(numberType, 255)
			]), []);
		Assert.That(result, Is.EqualTo(new ValueInstance(colorValueType, [
			new ValueInstance(numberType, 0), new ValueInstance(numberType, 0.5),
			new ValueInstance(numberType, 1)
		])));
	}

	[Test]
	[Benchmark]
	public void RunAllTestsInPackage() => interpreter.RunAllTestsInPackage(TestPackage.Instance);

	[Test]
	[Benchmark]
	public async Task RunAllTestsForAllStrictFilesInThisRepo()
	{
		var repos = new Repositories(new MethodExpressionParser());
		var strict = await repos.LoadStrictPackage();
		var math = await repos.LoadStrictPackage("Strict/Math");
		var imageProcessing = await repos.LoadStrictPackage("Strict/ImageProcessing");
		var language = await repos.LoadStrictPackage("Strict/Language");
		var expressions = await repos.LoadStrictPackage("Strict/Expressions");
		var examples = await repos.LoadStrictPackage("Strict/Examples");
		var fullInterpreter = new TestInterpreter(strict);
		var packages = new[] { strict, math, imageProcessing, language, expressions, examples };
		var tasks = new List<Task>();
		foreach (var packageToTest in packages)
			tasks.Add(Task.Run(() => fullInterpreter.RunAllTestsInPackage(packageToTest)));
		await Task.WhenAll(tasks);
		Console.WriteLine("All tests ran: " + fullInterpreter.Statistics);
	}

	//ncrunch: no coverage start
	[Test]
	[Category("Slow")]
	public void AllocatesLessThan40KbPerRunAfterWarmup()
	{
		interpreter.RunAllTestsInPackage(TestPackage.Instance);
		var allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
		interpreter.RunAllTestsInPackage(TestPackage.Instance);
		Assert.That(GC.GetAllocatedBytesForCurrentThread() - allocatedBefore, Is.LessThan(40_000));
	}

	[Test]
	[Category("Slow")]
	public void RunAllTestsInPackageTwice()
	{
		interpreter.RunAllTestsInPackage(TestPackage.Instance);
		interpreter.RunAllTestsInPackage(TestPackage.Instance);
	}

	[Test]
	[Category("Manual")]
	public void BenchmarkCompare() => BenchmarkRunner.Run<TestInterpreterTests>();
}