using Strict.Expressions;
using Strict.HighLevelRuntime;
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
	public Runner(string strictFilePath, bool enableDetailedOutput = false)
	{
		this.enableDetailedOutput = enableDetailedOutput;
		Log("╔════════════════════════════════════╗");
		Log("║ Strict Programming Language Runner ║");
		Log("╚════════════════════════════════════╝");
		Log("┌─ Step 1: Loading Strict package");
		var startTicks = DateTime.UtcNow.Ticks;
		var basePackage = TestPackage.Instance;
		var endTicks = DateTime.UtcNow.Ticks;
		Log("└─ Step 1 ⏱ Time: " +
			TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds + " ms");
		Log("┌─ Step 2: Create package and load type: " + strictFilePath);
		var startLoadTypeTicks = DateTime.UtcNow.Ticks;
		package = new Package(basePackage,
			Path.GetDirectoryName(Path.GetFullPath(strictFilePath)) ??
			throw new InvalidOperationException("Cannot determine package path"));
		var typeName = Path.GetFileNameWithoutExtension(strictFilePath);
		var typeLines = new TypeLines(typeName, File.ReadAllLines(strictFilePath));
		mainType = new Type(package, typeLines).ParseMembersAndMethods(new MethodExpressionParser());
		var endLoadTypeTicks = DateTime.UtcNow.Ticks;
		stepTimes.Add(endLoadTypeTicks - startLoadTypeTicks);
		Log("│  ✓ Loaded package: " + package.Name);
		Log("│  ✓ Created type: " + mainType.Name);
		Log("│  ✓ Members: " + mainType.Members.Count);
		Log("│  ✓ Methods: " + mainType.Methods.Count);
		Log("└─ Step 2 ⏱ Time: " +
			TimeSpan.FromTicks(endLoadTypeTicks - startLoadTypeTicks).TotalMilliseconds + " ms");
	}
	
	private readonly bool enableDetailedOutput;

	private void Log(string message)
	{
		if (enableDetailedOutput)
			Console.WriteLine(message); //ncrunch: no coverage
	}
	
	private readonly List<long> stepTimes = new();
	private readonly Package package;
	private readonly Type mainType;

	public void Run()
	{
		Parse();
		Validate();
		RunTests();
		//TODO: rename to instructions here too, we are really on a assembly level here and not high level statements anymore, this is just confusing
		var statements = GenerateBytecode();
		var optimizedStatements = OptimizeBytecode(statements);
		ExecuteBytecode(optimizedStatements);
		Log("⏱ Total Time (without Step 1 Loading Strict package): " +
			TimeSpan.FromTicks(stepTimes.Sum()).TotalMilliseconds + " ms");
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

	private List<Statement> GenerateBytecode()
	{
		Log("┌─ Step 6: Generate Bytecode");
		var startTicks = DateTime.UtcNow.Ticks;
		var statements = new List<Statement>();
		foreach (var method in mainType.Methods.Where(m => !m.IsTrait))
			try
			{
				//TODO: why is this so low level, can't we just generate the whole type/whole package?
				var arguments = method.Parameters.Select(
					//ncrunch: no coverage start
					p => (Expression)new Value(p.Type, new ValueInstance(p.Type, 0))).ToList();
				//ncrunch: no coverage end
				var methodCall = new MethodCall(method, null, arguments);
				var generator = new ByteCodeGenerator(methodCall);
				statements.AddRange(generator.Generate());
			} //ncrunch: no coverage start
			catch (Exception ex)
			{
				Log("│  ⚠ Failed method " + method.Name + ": " + ex.Message);
			} //ncrunch: no coverage end
		var endTicks = DateTime.UtcNow.Ticks;
		Log("│  ✓ Generated bytecode instructions: " + statements.Count);
		stepTimes.Add(endTicks - startTicks);
		Log("└─ Step 6 ⏱ Time: " +
			TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds + " ms");
		return statements;
	}

	private List<Statement> OptimizeBytecode(List<Statement> statements)
	{
		Log("┌─ Step 7: Optimize");
		var startTicks = DateTime.UtcNow.Ticks;
		var optimizedStatements = new List<Statement>(statements);
		var optimizers = new InstructionOptimizer[]
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
				Log($"│  ✓ {optimizer.GetType().Name}: removed {removed} instructions");
		}
		var endTicks = DateTime.UtcNow.Ticks;
		Log($"│  ✓ Total optimization: From {
			statements.Count
		} to {
			optimizedStatements.Count
		} instructions ({
			statements.Count - optimizedStatements.Count
		} removed, " + (statements.Count - optimizedStatements.Count) * 100 / statements.Count + "%)");
		stepTimes.Add(endTicks - startTicks);
		Log("└─ Step 7 ⏱ Time: " +
			TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds + " ms");
		return optimizedStatements;
	}

	private void ExecuteBytecode(List<Statement> statements) //TODO: use the actual statements
	{
		Log("┌─ Step 8: Execute");
		Log("│  ✓ Executing " + mainType.Name + ".Run method:");
		var startTicks = DateTime.UtcNow.Ticks;
		//TODO: this is completely wrong, itshould execute bytecode and not run the HighLevelRuntime interpreter!
		new Executor(package, TestBehavior.Disabled).ExecuteRunMethod(mainType);
		var endTicks = DateTime.UtcNow.Ticks;
		Log("│  ✓ Run method executed successfully");
		stepTimes.Add(endTicks - startTicks);
		Log("└─ Step 8 ⏱ Time: " +
			TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds + " ms");
	}
}