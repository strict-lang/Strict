using Strict.Expressions;
using Strict.Language;
using Strict.Language.Tests;
using Strict.Optimizers;
using Strict.Runtime;
using Strict.Runtime.Statements;
using Strict.TestRunner;
using Strict.Validators;
using Type = Strict.Language.Type;

namespace Strict;

public sealed class Runner
{
	//ncrunch: no coverage start
	public Runner(string strictFilePath)
	{
		Console.WriteLine("╔════════════════════════════════════╗");
		Console.WriteLine("║ Strict Programming Language Runner ║");
		Console.WriteLine("╚════════════════════════════════════╝");
		Console.WriteLine("┌─ Step 1: Loading Strict package");
		var startTicks = DateTime.UtcNow.Ticks;
		var basePackage = TestPackage.Instance;
		var endTicks = DateTime.UtcNow.Ticks;
		Console.WriteLine("└─ Step 1 ⏱ Time: " +
			TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds + " ms");
		Console.WriteLine("┌─ Step 2: Create package and load type: " + strictFilePath);
		var startLoadTypeTicks = DateTime.UtcNow.Ticks;
		package = new Package(basePackage,
			Path.GetDirectoryName(Path.GetFullPath(strictFilePath)) ??
			throw new InvalidOperationException("Cannot determine package path"));
		var typeName = Path.GetFileNameWithoutExtension(strictFilePath);
		var typeLines = new TypeLines(typeName, File.ReadAllLines(strictFilePath));
		mainType = new Type(package, typeLines).ParseMembersAndMethods(new MethodExpressionParser());
		var endLoadTypeTicks = DateTime.UtcNow.Ticks;
		stepTimes.Add(endLoadTypeTicks - startLoadTypeTicks);
		Console.WriteLine("│  ✓ Loaded package: " + package.Name);
		Console.WriteLine("│  ✓ Created type: " + mainType.Name);
		Console.WriteLine("│  ✓ Members: " + mainType.Members.Count);
		Console.WriteLine("│  ✓ Methods: " + mainType.Methods.Count);
		Console.WriteLine("└─ Step 2 ⏱ Time: " +
			TimeSpan.FromTicks(endLoadTypeTicks - startLoadTypeTicks).TotalMilliseconds + " ms");
	}

	private readonly List<long> stepTimes = new();
	private readonly Package package;
	private readonly Type mainType;

	public void Run()
	{
		Parse();
		Validate();
		RunTests();
		GenerateBytecode();
		OptimizeBytecode();
		ExecuteBytecode();
		Console.WriteLine("⏱ Total Time (without Step 1 Loading Strict package): " +
			TimeSpan.FromTicks(stepTimes.Sum()).TotalMilliseconds + " ms");
	}

	private void Parse()
	{
		Console.WriteLine("┌─ Step 3: Parse Method Bodies");
		var startTicks = DateTime.UtcNow.Ticks;
		var parsedMethods = 0;
		var totalExpressions = 0;
		foreach (var method in mainType.Methods)
		{
			if (method.IsTrait)
				continue;
			var body = method.GetBodyAndParseIfNeeded();
			parsedMethods++;
			if (body is Body bodyExpr)
				totalExpressions += bodyExpr.Expressions.Count;
			else
				totalExpressions++;
		}
		Console.WriteLine("│  ✓ Parsed methods: " + parsedMethods);
		Console.WriteLine("│  ✓ Total expressions: " + totalExpressions);
		var endTicks = DateTime.UtcNow.Ticks;
		stepTimes.Add(endTicks - startTicks);
		Console.WriteLine("└─ Step 3 ⏱ Time: " +
			TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds + " ms");
	}

	private void Validate()
	{
		Console.WriteLine("┌─ Step 4: Run Validators");
		var startTicks = DateTime.UtcNow.Ticks;
		try
		{
			new TypeValidator().Visit(mainType);
			Console.WriteLine("│  ✓ All type validations passed, no unused expressions found");
			var constants = new ConstantCollapser();
			constants.Visit(mainType);
			Console.WriteLine("│  ✓ Constant expressions collapsed: " + constants.CollapsedCount);
		}
		catch (Exception ex)
		{
			Console.WriteLine("│  ✗ Validation failed: " + ex.Message);
			throw;
		}
		var endTicks = DateTime.UtcNow.Ticks;
		stepTimes.Add(endTicks - startTicks);
		Console.WriteLine("└─ Step 4 ⏱ Time: " +
			TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds + " ms");
	}

	private void RunTests()
	{
		Console.WriteLine("┌─ Step 5: Run Tests");
		var startTicks = DateTime.UtcNow.Ticks;
		var testExecutor = new TestExecutor(package);
		try
		{
			testExecutor.RunAllTestsInType(mainType);
			Console.WriteLine("│  ✓ Methods tested: " + testExecutor.Statistics.MethodsTested);
			Console.WriteLine("│  ✓ Types tested: " + testExecutor.Statistics.TypesTested);
			Console.WriteLine("│  ✓ " + testExecutor.Statistics);
			Console.WriteLine("│  ✓ All tests passed");
		}
		catch (Exception ex)
		{
			Console.WriteLine($"│  ✗ Tests failed: {ex.Message}");
			throw;
		}
		var endTicks = DateTime.UtcNow.Ticks;
		stepTimes.Add(endTicks - startTicks);
		Console.WriteLine("└─ Step 5 ⏱ Time: " +
			TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds + " ms");
	}

	private List<Statement>? generatedStatements;

	private void GenerateBytecode()
	{
		Console.WriteLine("┌─ Step 6: Generate Bytecode");
		var startTicks = DateTime.UtcNow.Ticks;
		var totalInstructions = 0;
		generatedStatements = [];
		foreach (var method in mainType.Methods.Where(m => !m.IsTrait))
			try
			{
				var arguments = method.Parameters.Select(p =>
					(Expression)new Value(p.Type, new ValueInstance(p.Type, 0))).ToList();
				var methodCall = new MethodCall(method, null, arguments);
				var generator = new ByteCodeGenerator(methodCall);
				var statements = generator.Generate();
				generatedStatements.AddRange(statements);
				totalInstructions += statements.Count;
			}
			catch (Exception ex)
			{
				Console.WriteLine("│  ⚠ Failed method " + method.Name + ": " + ex.Message);
			}
		var endTicks = DateTime.UtcNow.Ticks;
		Console.WriteLine("│  ✓ Generated bytecode instructions: " + totalInstructions);
		stepTimes.Add(endTicks - startTicks);
		Console.WriteLine("└─ Step 6 ⏱ Time: " +
			TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds + " ms");
	}

	private List<Statement>? optimizedStatements;

	private void OptimizeBytecode()
	{
		Console.WriteLine("┌─ Step 7: Optimize");
		if (generatedStatements == null || generatedStatements.Count == 0)
		{
			Console.WriteLine("│  ⚠ No bytecode to optimize");
			Console.WriteLine("└─ Step 7 Skipped");
			return;
		}
		var startTicks = DateTime.UtcNow.Ticks;
		var originalCount = generatedStatements.Count;
		optimizedStatements = new List<Statement>(generatedStatements);
		var optimizers = new StatementOptimizer[]
		{
			new TestCodeRemover(), new ConstantFoldingOptimizer(), new DeadStoreEliminator(),
			new UnreachableCodeEliminator(), new RedundantLoadEliminator()
		};
		foreach (var optimizer in optimizers)
		{
			var beforeCount = optimizedStatements.Count;
			optimizedStatements = optimizer.Optimize(optimizedStatements);
			var removed = beforeCount - optimizedStatements.Count;
			if (removed > 0)
				Console.WriteLine($"│  ✓ {optimizer.GetType().Name}: removed {removed} instructions");
		}
		var endTicks = DateTime.UtcNow.Ticks;
		Console.WriteLine($"│  ✓ Total optimization: From {
			originalCount
		} to {
			optimizedStatements.Count
		} instructions ({
			originalCount - optimizedStatements.Count
		} removed, " + (originalCount - optimizedStatements.Count) * 100 / originalCount + "%)");
		stepTimes.Add(endTicks - startTicks);
		Console.WriteLine("└─ Step 7 ⏱ Time: " +
			TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds + " ms");
	}

	private void ExecuteBytecode()
	{
		Console.WriteLine("┌─ Step 8: Execute");
		var startTicks = DateTime.UtcNow.Ticks;
		if (optimizedStatements == null || optimizedStatements.Count == 0)
		{
			Console.WriteLine("│  ⚠ No optimized bytecode to execute");
			Console.WriteLine("└─ Step 8 Skipped");
			return;
		}
		Console.WriteLine(
			"│  ℹ Bytecode execution requires a proper entry point with instance context");
		Console.WriteLine("│  ℹ Generated bytecode statements: " + optimizedStatements.Count);
		Console.WriteLine(
			"│  ℹ For actual execution, use Strict.TestRunner which handles instance creation");
		var endTicks = DateTime.UtcNow.Ticks;
		stepTimes.Add(endTicks - startTicks);
		Console.WriteLine("└─ Step 8 ⏱ Time: " +
			TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds + " ms");
		//TODO: run actual code and give the result here!
	}
}