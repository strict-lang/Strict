using Strict.Expressions;
using Strict.Language;
using Strict.Optimizers;
using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.TestRunner;
using Strict.Validators;
using Type = Strict.Language.Type;

namespace Strict;

public sealed class Runner : IDisposable
{
	/// <summary>
	/// Creates a Runner for a .strict source file (parses, validates, compiles, and runs it),
	/// or for a .strict_binary ZIP file (loads pre-compiled bytecode and executes it directly).
	/// </summary>
	public Runner(Package basePackage, string strictFilePath,
		bool enableTestsAndDetailedOutput = false)
	{
		this.enableTestsAndDetailedOutput = enableTestsAndDetailedOutput;
		Log("╔════════════════════════════════════╗");
		Log("║ Strict Programming Language Runner ║");
		Log("╚════════════════════════════════════╝");
		Log("┌─ Step 1: Loading: " + strictFilePath);
		var startTicks = DateTime.UtcNow.Ticks;
		currentFolder = Path.GetDirectoryName(Path.GetFullPath(strictFilePath))!;
		var typeName = Path.GetFileNameWithoutExtension(strictFilePath);
		if (Path.GetExtension(strictFilePath) == BytecodeSerializer.Extension)
		{
			deserializer = new BytecodeDeserializer(strictFilePath, basePackage);
			package = deserializer.Package;
			mainType = package.GetType(typeName);
		}
		else
		{
			package = new Package(basePackage,
				Path.GetDirectoryName(Path.GetFullPath(strictFilePath)) ??
				throw new InvalidOperationException("Cannot determine package path"));
			var typeLines = new TypeLines(typeName, File.ReadAllLines(strictFilePath));
			mainType =
				new Type(package, typeLines).ParseMembersAndMethods(new MethodExpressionParser());
		}
		var endTicks = DateTime.UtcNow.Ticks;
		stepTimes.Add(endTicks - startTicks);
		Log("└─ Step 1 ⏱ Time: " +
			TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds + " ms");
	}

	private readonly bool enableTestsAndDetailedOutput;
	private readonly string currentFolder;
	private readonly BytecodeDeserializer? deserializer;
	private readonly Package package;
	private readonly Type mainType;
	private readonly List<long> stepTimes = new();

	private void Log(string message)
	{
		if (enableTestsAndDetailedOutput)
			Console.WriteLine(message);
	}

	public Runner Run() =>
		deserializer != null
			? RunFromPreloadedBytecode(deserializer.Instructions[mainType.Name])
			: RunFromSource();

	private Runner RunFromPreloadedBytecode(List<Instruction> preloadedInstructions)
	{
		Log("╔═══════════════════════════════════════════╗");
		Log("║  Running from pre-compiled .strictbinary  ║");
		Log("╚═══════════════════════════════════════════╝");
		ExecuteBytecode(preloadedInstructions);
		Log("Successfully executed pre-compiled " + mainType.Name + " in " +
			TimeSpan.FromTicks(stepTimes.Sum()).ToString(@"s\.ffffff") + "s");
		return this;
	}

	private Runner RunFromSource()
	{
		if (enableTestsAndDetailedOutput)
		{
			Parse();
			Validate();
			RunTests();
		}
		var instructions = GenerateBytecode();
		var optimizedInstructions = OptimizeBytecode(instructions);
		ExecuteBytecode(optimizedInstructions);
		SaveBytecodeIfPossible(optimizedInstructions);
		Console.WriteLine("Successfully parsed, optimized and executed " + mainType.Name + " in " +
			TimeSpan.FromTicks(stepTimes.Sum()).ToString(@"s\.ffffff") + "s");
		return this;
	}

	private void Parse()
	{
		Log("┌─ Step 3: Parse Method Bodies");
		var startTicks = DateTime.UtcNow.Ticks;
		var parsedMethods = 0;
		var totalExpressions = 0;
		foreach (var method in mainType.Methods)
			if (!method.IsTrait)
			{
				var body = method.GetBodyAndParseIfNeeded();
				parsedMethods++;
				if (body is Body bodyExpr)
					totalExpressions += bodyExpr.Expressions.Count;
				else
					totalExpressions++; //ncrunch: no coverage
			}
		Log("│  ✓ Parsed methods: " + parsedMethods);
		Log("│  ✓ Total expressions: " + totalExpressions);
		var endTicks = DateTime.UtcNow.Ticks;
		stepTimes.Add(endTicks - startTicks);
		Log("└─ Step 3 ⏱ Time: " +
			TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds + " ms");
	}

	private void Validate()
	{
		Log("┌─ Step 4: Run Validators");
		var startTicks = DateTime.UtcNow.Ticks;
		try
		{
			new TypeValidator().Visit(mainType);
			Log("│  ✓ All type validations passed, no unused expressions found");
			var constants = new ConstantCollapser();
			constants.Visit(mainType);
			Log("│  ✓ Constant expressions collapsed: " + constants.CollapsedCount);
		}
		//ncrunch: no coverage start
		catch (Exception ex)
		{
			Log("│  ✗ Validation failed: " + ex.Message);
			throw;
		} //ncrunch: no coverage end
		var endTicks = DateTime.UtcNow.Ticks;
		stepTimes.Add(endTicks - startTicks);
		Log("└─ Step 4 ⏱ Time: " +
			TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds + " ms");
	}

	private void RunTests()
	{
		Log("┌─ Step 5: Run Tests");
		var startTicks = DateTime.UtcNow.Ticks;
		var testExecutor = new TestInterpreter(package);
		try
		{
			testExecutor.RunAllTestsInType(mainType);
			Log("│  ✓ Methods tested: " + testExecutor.Statistics.MethodsTested);
			Log("│  ✓ Types tested: " + testExecutor.Statistics.TypesTested);
			Log("│  ✓ " + testExecutor.Statistics);
			Log("│  ✓ All tests passed");
		}
		//ncrunch: no coverage start
		catch (Exception ex)
		{
			Log($"│  ✗ Tests failed: {ex.Message}");
			throw;
		} //ncrunch: no coverage end
		var endTicks = DateTime.UtcNow.Ticks;
		stepTimes.Add(endTicks - startTicks);
		Log("└─ Step 5 ⏱ Time: " +
			TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds + " ms");
	}

	private List<Instruction> GenerateBytecode()
	{
		Log("┌─ Step 6: Generate Bytecode");
		var startTicks = DateTime.UtcNow.Ticks;
		var runMethod =
			mainType.Methods.FirstOrDefault(m => m is { Name: Method.Run, Parameters.Count: 0 }) ??
			throw new InvalidOperationException("No Run method found on " + mainType.Name);
		var runMethodCall = new MethodCall(runMethod, null, Array.Empty<Expression>());
		var instructions = new BytecodeGenerator(runMethodCall).Generate();
		var endTicks = DateTime.UtcNow.Ticks;
		Log("│  ✓ Generated bytecode instructions: " + instructions.Count);
		stepTimes.Add(endTicks - startTicks);
		Log("└─ Step 6 ⏱ Time: " +
			TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds + " ms");
		return instructions;
	}

	private List<Instruction> OptimizeBytecode(List<Instruction> instructions)
	{
		Log("┌─ Step 7: Optimize");
		var startTicks = DateTime.UtcNow.Ticks;
		var optimizedInstructions = new List<Instruction>(instructions);
		var optimizers = new InstructionOptimizer[]
		{
			new TestCodeRemover(), new ConstantFoldingOptimizer(), new DeadStoreEliminator(),
			new UnreachableCodeEliminator(), new RedundantLoadEliminator()
		};
		foreach (var optimizer in optimizers)
		{
			var beforeCount = optimizedInstructions.Count;
			optimizedInstructions = optimizer.Optimize(optimizedInstructions);
			var removed = beforeCount - optimizedInstructions.Count;
			if (removed > 0)
				Log($"│  ✓ {optimizer.GetType().Name}: removed {removed} instructions"); //ncrunch: no coverage
		}
		var endTicks = DateTime.UtcNow.Ticks;
		Log($"│  ✓ Total optimization: From {
			instructions.Count
		} to {
			optimizedInstructions.Count
		} instructions ({
			instructions.Count - optimizedInstructions.Count
		} removed, " + (instructions.Count - optimizedInstructions.Count) * 100 / instructions.Count + "%)");
		stepTimes.Add(endTicks - startTicks);
		Log("└─ Step 7 ⏱ Time: " +
			TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds + " ms");
		return optimizedInstructions;
	}

	private void ExecuteBytecode(List<Instruction> instructions)
	{
		Log("┌─ Step 8: Execute");
		Log("│  ✓ Executing " + mainType.Name + ".Run method:");
		var startTicks = DateTime.UtcNow.Ticks;
		new VirtualMachine(package).Execute(instructions);
		var endTicks = DateTime.UtcNow.Ticks;
		Log("│  ✓ Run method executed successfully, instructions: " + instructions.Count);
		stepTimes.Add(endTicks - startTicks);
		Log("└─ Step 8 ⏱ Time: " + TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds +
			" ms");
	}

	private void SaveBytecodeIfPossible(List<Instruction> optimizedInstructions)
	{
		var serializer = new BytecodeSerializer(
			new Dictionary<string, IList<Instruction>> { [mainType.Name] = optimizedInstructions },
			currentFolder, mainType.Name);
		Console.WriteLine("Saving " + new FileInfo(serializer.OutputFilePath).Length +
			" bytes of bytecode to: " + serializer.OutputFilePath);
	}

	public void Dispose() => package.Dispose();
}