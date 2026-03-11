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
	/// Creates a Runner for a .strict source file (parses, validates, compiles and runs it),
	/// or for a .strict_binary ZIP file (loads pre-compiled bytecode and executes it directly).
	/// </summary>
	public Runner(Package basePackage, string strictFilePath,
		bool enableTestsAndDetailedOutput = false)
	{
		this.enableTestsAndDetailedOutput = enableTestsAndDetailedOutput;
		Log("╔════════════════════════════════════╗");
		Log("║ Strict Programming Language Runner ║");
		Log("╚════════════════════════════════════╝");
		try
		{
			if (Path.GetExtension(strictFilePath) == BytecodeSerializer.Extension)
				InitializeFromBinaryFile(basePackage, strictFilePath);
			else
				InitializeFromSourceFile(basePackage, strictFilePath);
		}
		catch
		{
			if (package != null)
				package.Dispose();
			throw;
		}
	}

	private void InitializeFromBinaryFile(Package basePackage, string binaryFilePath)
	{
		Log("┌─ Step 1: Loading pre-compiled binary: " + binaryFilePath);
		var startTicks = DateTime.UtcNow.Ticks;
		var binaryDir = Path.GetDirectoryName(Path.GetFullPath(binaryFilePath)) ?? ".";
		package = new Package(basePackage, binaryDir);
		BytecodeSerializer.LoadEmbeddedTypes(binaryFilePath, package);
		var requestedTypeName = Path.GetFileNameWithoutExtension(binaryFilePath);
		var bytecodeByType = BytecodeSerializer.DeserializeAll(binaryFilePath, package);
		var selectedTypeName = requestedTypeName;
		if (!bytecodeByType.TryGetValue(selectedTypeName, out preloadedBytecode))
		{
			selectedTypeName = bytecodeByType.Keys.First();
			preloadedBytecode = bytecodeByType[selectedTypeName];
		}
		mainType = package.GetType(selectedTypeName);
		var endTicks = DateTime.UtcNow.Ticks;
		stepTimes.Add(endTicks - startTicks);
		Log("└─ Step 1 ⏱ Time: " +
			TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds + " ms");
	}

	private void InitializeFromSourceFile(Package basePackage, string strictFilePath)
	{
		Log("┌─ Step 1: Create package and load type: " + strictFilePath);
		var startLoadTypeTicks = DateTime.UtcNow.Ticks;
		package = new Package(basePackage,
			Path.GetDirectoryName(Path.GetFullPath(strictFilePath)) ??
			throw new InvalidOperationException("Cannot determine package path"));
		var typeName = Path.GetFileNameWithoutExtension(strictFilePath);
		var typeLines = new TypeLines(typeName, File.ReadAllLines(strictFilePath));
		mainType = new Type(package, typeLines).ParseMembersAndMethods(new MethodExpressionParser());
		var endLoadTypeTicks = DateTime.UtcNow.Ticks;
		stepTimes.Add(endLoadTypeTicks - startLoadTypeTicks);
		sourceFilePath = strictFilePath;
		Log("│  ✓ Loaded package: " + package.Name);
		Log("│  ✓ Created type: " + mainType.Name);
		Log("│  ✓ Members: " + mainType.Members.Count);
		Log("│  ✓ Methods: " + mainType.Methods.Count);
		Log("└─ Step 1 ⏱ Time: " +
			TimeSpan.FromTicks(endLoadTypeTicks - startLoadTypeTicks).TotalMilliseconds + " ms");
	}

	private readonly bool enableTestsAndDetailedOutput;
	private List<Instruction>? preloadedBytecode;
	private string? sourceFilePath;

	private void Log(string message)
	{
		if (enableTestsAndDetailedOutput)
			Console.WriteLine(message);
	}

	private readonly List<long> stepTimes = new();
	private Package package = null!;
	private Type mainType = null!;

	public Runner Run() =>
		preloadedBytecode != null
			? RunFromPreloadedBytecode()
			: RunFromSource();

	private Runner RunFromPreloadedBytecode()
	{
		Log("╔═══════════════════════════════════════════╗");
		Log("║  Running from pre-compiled .strictbinary  ║");
		Log("╚═══════════════════════════════════════════╝");
		ExecuteBytecode(preloadedBytecode!);
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
		SaveBytecodeIfPossible(optimizedInstructions);
		ExecuteBytecode(optimizedInstructions);
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
		var testExecutor = new TestExecutor(package);
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
		var binaryPath = Path.ChangeExtension(sourceFilePath, BytecodeSerializer.Extension)!;
		Log("┌─ Saving bytecode to: " + binaryPath);
		BytecodeSerializer.Serialize(optimizedInstructions, binaryPath, mainType.Name,
			Path.GetDirectoryName(sourceFilePath!));
		Log("└─ Bytecode saved: " + new FileInfo(binaryPath).Length + " bytes");
	}

	public void Dispose() => package.Dispose();
}