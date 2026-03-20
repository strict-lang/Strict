using Type = Strict.Language.Type;

namespace Strict.Bytecode.Tests;

public sealed class DecompilerTests : TestBytecode
{
	[Test]
	public void DecompileSimpleArithmeticBytecodeCreatesStrictFile()
	{
		var instructions = new BinaryGenerator(
			GenerateMethodCallFromSource("Add", "Add(10, 5).Calculate",
				"has First Number", "has Second Number", "Calculate Number",
				"\tAdd(10, 5).Calculate is 15", "\tFirst + Second")).Generate();
		var outputFolder = DecompileToTemp(instructions, "Add");
		try
		{
			var content = File.ReadAllText(Path.Combine(outputFolder, "Add.strict"));
			Assert.That(content, Does.Contain("First + Second"));
		}
		finally
		{
			if (Directory.Exists(outputFolder))
				Directory.Delete(outputFolder, recursive: true);
		}
	}

	[Test]
	public void DecompileRunMethodReconstructsConstantDeclarationFromMethodCall()
	{
		var instructions = new BinaryGenerator(
			GenerateMethodCallFromSource("Counter", "Counter(5).Calculate",
				"has count Number",
				"Double Number",
				"\tCounter(3).Double is 6",
				"\tcount * 2",
				"Calculate Number",
				"\tCounter(5).Calculate is 10",
				"\tconstant doubled = Counter(3).Double",
				"\tdoubled * 2")).Generate();
		var outputFolder = DecompileToTemp(instructions, "Counter");
		try
		{
			var content = File.ReadAllText(Path.Combine(outputFolder, "Counter.strict"));
			Assert.That(content, Does.Contain("Run"));
			Assert.That(content, Does.Contain("Counter(3).Double"));
		}
		finally
		{
			if (Directory.Exists(outputFolder))
				Directory.Delete(outputFolder, recursive: true);
		}
	}

	[Test]
	public void DecompileSumExampleContainsLoggerCall()
	{
		var outputFolder = DecompileExampleToTemp("Sum");
		try
		{
			var content = File.ReadAllText(Path.Combine(outputFolder, "Sum.strict"));
			Assert.That(content, Does.Contain("logger.Log"));
		}
		finally
		{
			if (Directory.Exists(outputFolder))
				Directory.Delete(outputFolder, recursive: true);
		}
	}

	[Test]
	public void DecompileSimpleCalculatorExampleContainsAddAndMultiplyBodies()
	{
		var outputFolder = DecompileExampleToTemp("SimpleCalculator");
		try
		{
			var content = File.ReadAllText(Path.Combine(outputFolder, "SimpleCalculator.strict"));
			Assert.That(content, Does.Contain("first + second"));
			Assert.That(content, Does.Contain("first * second"));
			Assert.That(content, Does.Not.Contain("Strict/"), "Type names should use short form");
			var lines = content.Split('\n', StringSplitOptions.RemoveEmptyEntries);
			var lastMethodLine = Array.FindLastIndex(lines, line => !line.StartsWith('\t'));
			Assert.That(lines[lastMethodLine], Does.StartWith("Run"), "Run method must be last");
			var runStart = Array.FindIndex(lines,
				line => line.TrimStart().StartsWith("Run", StringComparison.Ordinal));
			var runBody = lines.Skip(runStart + 1).TakeWhile(line => line.StartsWith('\t')).ToArray();
			Assert.That(runBody, Has.None.EqualTo("\tcalc.Multiply"),
				"Run body must not contain spurious calc.Multiply expression");
		}
		finally
		{
			if (Directory.Exists(outputFolder))
				Directory.Delete(outputFolder, recursive: true);
		}
	}

	[Test]
	public void DecompileConstantMemberWritesConstKeywordWithValue()
	{
		var parser = new MethodExpressionParser();
		var package = new Package(TestPackage.Instance, "BConstTest");
		var type = new Type(package, new TypeLines("Score",
			["constant Minimum = 0", "constant Maximum = 100", "has value Number",
				"Run", "\tMinimum is 0", "\tMaximum is 100"])).ParseMembersAndMethods(parser);
		var runMethods = type.Methods.Where(method => method.Name == Method.Run).ToArray();
		var binary = BinaryGenerator.GenerateFromRunMethods(runMethods[0], runMethods);
		var outputFolder = DecompileToTemp(binary, "Score");
		try
		{
			var content = File.ReadAllText(Path.Combine(outputFolder, "Score.strict"));
			Assert.That(content, Does.Contain("const Minimum = 0"), "Constant must use const keyword with value");
			Assert.That(content, Does.Contain("const Maximum = 100"), "Constant must use const keyword with value");
			Assert.That(content, Does.Not.Contain("has Minimum"), "Constants must not use has keyword");
			Assert.That(content, Does.Contain("has value Number"), "Regular members must still use has");
		}
		finally
		{
			if (Directory.Exists(outputFolder))
				Directory.Delete(outputFolder, recursive: true);
		}
	}

	[Test]
	public void DecompileRemoveParenthesesExampleContainsLoopAndConditions()
	{
		var outputFolder = DecompileExampleToTemp("RemoveParentheses");
		try
		{
			var content = File.ReadAllText(Path.Combine(outputFolder, "RemoveParentheses.strict"));
			Assert.That(content, Does.Contain("for"));
			Assert.That(content, Does.Contain("if"));
		}
		finally
		{
			if (Directory.Exists(outputFolder))
				Directory.Delete(outputFolder, recursive: true);
		}
	}

	private static string DecompileExampleToTemp(string typeName) =>
		DecompileToTemp(GenerateBinaryFromExample(typeName), typeName);

	private static BinaryExecutable GenerateBinaryFromExample(string typeName)
	{
		var parser = new MethodExpressionParser();
		var package = new Package(TestPackage.Instance, "B" + typeName[..Math.Min(typeName.Length, 10)]);
		var strictFilePath = Path.Combine(GetExamplesFolder(), typeName + Type.Extension);
		var sourceLines = File.ReadAllLines(strictFilePath);
		if (typeName == "RemoveParentheses")
			sourceLines = sourceLines.Select(line => line.Replace(".Increase", ".Increment").
				Replace(".Decrease", ".Decrement")).ToArray();
		var type = new Type(package, new TypeLines(typeName, sourceLines)).ParseMembersAndMethods(parser);
		var runMethods = type.Methods.Where(method => method.Name == Method.Run).ToArray();
		if (runMethods.Length > 0)
			return BinaryGenerator.GenerateFromRunMethods(runMethods[0], runMethods);
		var expression = (MethodCall)parser.ParseExpression(
			new Body(new Method(type, 0, parser, [nameof(GenerateBinaryFromExample)])),
			GetDefaultMethodCall(typeName));
		return new BinaryGenerator(expression).Generate();
	}

	private static string GetDefaultMethodCall(string typeName) =>
		typeName == "RemoveParentheses"
			? "RemoveParentheses(\"example(unwanted thing)example\").Remove"
			: typeName + ".Run";

	private static string GetExamplesFolder()
	{
		var path = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "Examples"));
		return Directory.Exists(path)
			? path
			: @"c:\code\GitHub\strict-lang\Strict\Examples";
	}

	private static string DecompileToTemp(BinaryExecutable strictBinary, string typeName)
	{
		var outputFolder = Path.Combine(Path.GetTempPath(), "decompiled_" + Path.GetRandomFileName());
		var targetOnlyBinary = new BinaryExecutable(TestPackage.Instance)
		{
			MethodsPerType =
			{
				[typeName] = strictBinary.MethodsPerType.First(method =>
					method.Key.EndsWith(typeName, StringComparison.Ordinal)).Value
			}
		};
		new Decompiler().Decompile(targetOnlyBinary, outputFolder);
		Assert.That(Directory.Exists(outputFolder), Is.True, "Output folder should be created");
		Assert.That(File.Exists(Path.Combine(outputFolder, typeName + ".strict")), Is.True,
			typeName + ".strict should be created");
		return outputFolder;
	}
}