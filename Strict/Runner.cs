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
		ExecuteBytecode(preloadedInstructions, deserializer?.PrecompiledMethods);
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

	private void ExecuteBytecode(List<Instruction> instructions,
		Dictionary<string, List<Instruction>>? precompiledMethods = null)
	{
		Log("┌─ Step 8: Execute");
		Log("│  ✓ Executing " + mainType.Name + ".Run method:");
		var startTicks = DateTime.UtcNow.Ticks;
		new VirtualMachine(package, precompiledMethods).Execute(instructions);
		var endTicks = DateTime.UtcNow.Ticks;
		Log("│  ✓ Run method executed successfully, instructions: " + instructions.Count);
		stepTimes.Add(endTicks - startTicks);
		Log("└─ Step 8 ⏱ Time: " + TimeSpan.FromTicks(endTicks - startTicks).TotalMilliseconds +
			" ms");
	}

	private void SaveBytecodeIfPossible(List<Instruction> optimizedInstructions)
	{
		var typeBytecodeData = BuildTypeBytecodeData(optimizedInstructions);
		var serializer = new BytecodeSerializer(typeBytecodeData, currentFolder, mainType.Name);
		Console.WriteLine("Saving " + new FileInfo(serializer.OutputFilePath).Length +
			" bytes of bytecode to: " + serializer.OutputFilePath);
	}

	private IReadOnlyList<TypeBytecodeData> BuildTypeBytecodeData(
		List<Instruction> optimizedRunInstructions)
	{
		var methodsByType = new Dictionary<Type, Dictionary<Method, IList<Instruction>>>();
		var methodsToCompile = new Queue<Method>();
		var compiledMethodKeys = new HashSet<string>(StringComparer.Ordinal);
		EnqueueCalledMethods(optimizedRunInstructions, methodsToCompile, compiledMethodKeys);
		while (methodsToCompile.Count > 0)
		{
			var method = methodsToCompile.Dequeue();
			if (method.IsTrait || !IsTypeInsideCurrentPackage(method.Type))
				continue;
			var methodExpressions = GetMethodExpressions(method);
			var methodInstructions = new BytecodeGenerator(
				new InvokedMethod(methodExpressions, EmptyArguments, method.ReturnType),
				new Registry()).Generate();
			if (!methodsByType.TryGetValue(method.Type, out var typeMethods))
			{
				typeMethods = new Dictionary<Method, IList<Instruction>>();
				methodsByType[method.Type] = typeMethods;
			}
			typeMethods[method] = methodInstructions;
			EnqueueCalledMethods(methodInstructions, methodsToCompile, compiledMethodKeys);
		}
		var requiredTypes = CollectRequiredTypes(methodsByType, optimizedRunInstructions);
		var usedMethodKeys = compiledMethodKeys;
		var calledTypeNames = usedMethodKeys.Select(methodKey => methodKey.Split('|')[0]).
			ToHashSet(StringComparer.Ordinal);
		var memberTypeNames = new HashSet<string>(mainType.Members.Select(member =>
			member.Type.Name), StringComparer.Ordinal);
		foreach (var currentPackageType in methodsByType.Keys.Where(IsTypeInsideCurrentPackage))
		foreach (var member in currentPackageType.Members)
			memberTypeNames.Add(member.Type.Name);
		var typeData = new List<TypeBytecodeData>();
		foreach (var requiredType in requiredTypes)
		{
			if (!IsTypeInsideCurrentPackage(requiredType) &&
				!calledTypeNames.Contains(requiredType.Name) &&
				!memberTypeNames.Contains(requiredType.Name))
				continue;
			methodsByType.TryGetValue(requiredType, out var compiledMethods);
			var methodDefinitions = requiredType.Methods.Where(method =>
				usedMethodKeys.Contains(BytecodeDeserializer.BuildMethodInstructionKey(requiredType.Name,
					method.Name, method.Parameters.Count)) ||
				requiredType == mainType && method.Name == Method.Run &&
				method.Parameters.Count == 0).Select(method =>
				new MethodBytecodeData(method.Name,
					method.Parameters.Select(parameter =>
						new MethodParameterBytecodeData(parameter.Name, parameter.Type.Name)).ToList(),
					method.ReturnType.Name)).ToList();
			var members = requiredType.Members.Where(member => !member.IsConstant).Select(member =>
				new MemberBytecodeData(member.Name, member.Type.Name)).ToList();
			var methodInstructions = new Dictionary<MethodBytecodeData, IList<Instruction>>();
			if (compiledMethods != null)
				foreach (var compiledMethod in compiledMethods)
					methodInstructions[new MethodBytecodeData(compiledMethod.Key.Name,
						compiledMethod.Key.Parameters.Select(parameter =>
							new MethodParameterBytecodeData(parameter.Name, parameter.Type.Name)).ToList(), //ncrunch: no coverage
						compiledMethod.Key.ReturnType.Name)] = compiledMethod.Value;
			typeData.Add(new TypeBytecodeData(requiredType.Name, BuildTypeEntryPath(requiredType),
				members, methodDefinitions,
				requiredType == mainType
					? optimizedRunInstructions
					: EmptyInstructions,
				methodInstructions));
		}
		return typeData;
	}

	private HashSet<Type> CollectRequiredTypes(
		Dictionary<Type, Dictionary<Method, IList<Instruction>>> methodsByType,
		IReadOnlyList<Instruction> optimizedRunInstructions)
	{
		var requiredTypes = new HashSet<Type> { mainType };
		foreach (var member in mainType.Members)
			requiredTypes.Add(member.Type);
		foreach (var instruction in optimizedRunInstructions)
			CollectInstructionTypes(instruction, requiredTypes);
		foreach (var methodEntry in methodsByType)
		{
			requiredTypes.Add(methodEntry.Key);
			if (IsTypeInsideCurrentPackage(methodEntry.Key))
				foreach (var member in methodEntry.Key.Members)
					requiredTypes.Add(member.Type);
			foreach (var compiledMethod in methodEntry.Value)
			{
				AddMethodSignatureTypes(compiledMethod.Key, requiredTypes);
				foreach (var instruction in compiledMethod.Value)
					CollectInstructionTypes(instruction, requiredTypes);
			}
		}
		return requiredTypes;
	}

	private static void AddMethodSignatureTypes(Method method, ISet<Type> requiredTypes)
	{
		requiredTypes.Add(method.Type);
		requiredTypes.Add(method.ReturnType);
		foreach (var parameter in method.Parameters)
			requiredTypes.Add(parameter.Type);
	}

	private static void CollectInstructionTypes(Instruction instruction, ISet<Type> requiredTypes)
	{
		if (instruction is not Invoke { Method: not null } invoke)
			return;
		AddMethodSignatureTypes(invoke.Method.Method, requiredTypes);
		requiredTypes.Add(invoke.Method.ReturnType);
	}

	private static readonly IList<Instruction> EmptyInstructions = Array.Empty<Instruction>();
	private static readonly IReadOnlyDictionary<Method, IList<Instruction>> EmptyMethodInstructions =
		new Dictionary<Method, IList<Instruction>>();

	private string BuildTypeEntryPath(Type type) =>
		IsTypeInsideCurrentPackage(type)
			? package.Name + "/" + type.Name
			: "Strict/" + type.Name;

	private bool IsTypeInsideCurrentPackage(Type type) =>
		type.Package == package ||
		type.Package.FullName.StartsWith(package.FullName + Context.ParentSeparator,
			StringComparison.Ordinal);

	private static IReadOnlyList<Expression> GetMethodExpressions(Method method)
	{
		var methodBody = method.GetBodyAndParseIfNeeded();
		return methodBody is Body body
			? body.Expressions
			: [methodBody];
	}

	// ReSharper disable once CollectionNeverUpdated.Local
	private static readonly Dictionary<string, ValueInstance> EmptyArguments =
		new(StringComparer.Ordinal);

	private static void EnqueueCalledMethods(IReadOnlyList<Instruction> instructions,
		Queue<Method> methodsToCompile, HashSet<string> compiledMethodKeys)
	{
		foreach (var invokeInstruction in instructions.OfType<Invoke>())
			if (invokeInstruction.Method != null)
			{
				EnqueueCalledMethod(invokeInstruction.Method.Method, methodsToCompile,
					compiledMethodKeys);
				if (invokeInstruction.Method.Instance != null)
					EnqueueMethodsFromExpression(invokeInstruction.Method.Instance, methodsToCompile,
						compiledMethodKeys);
				foreach (var argument in invokeInstruction.Method.Arguments)
					EnqueueMethodsFromExpression(argument, methodsToCompile, compiledMethodKeys);
			}
	}

	private static void EnqueueMethodsFromExpression(Expression expression,
		Queue<Method> methodsToCompile, HashSet<string> compiledMethodKeys)
	{
		switch (expression)
		{
		case MethodCall methodCall:
			EnqueueCalledMethod(methodCall.Method, methodsToCompile, compiledMethodKeys);
			if (methodCall.Instance != null)
				EnqueueMethodsFromExpression(methodCall.Instance, methodsToCompile,
					compiledMethodKeys);
			foreach (var argument in methodCall.Arguments)
				EnqueueMethodsFromExpression(argument, methodsToCompile, compiledMethodKeys);
			break;
		case MemberCall { Instance: not null } memberCall:
			// ReSharper disable once TailRecursiveCall
			//ncrunch: no coverage start
			EnqueueMethodsFromExpression(memberCall.Instance, methodsToCompile, compiledMethodKeys);
			break;
		case List listExpression:
			foreach (var value in listExpression.Values)
				EnqueueMethodsFromExpression(value, methodsToCompile, compiledMethodKeys);
			break; //ncrunch: no coverage end
		}
	}

	private static void EnqueueCalledMethod(Method method, Queue<Method> methodsToCompile,
		HashSet<string> compiledMethodKeys)
	{
		if (method.Name == Method.From)
			return;
		var methodKey = BytecodeDeserializer.BuildMethodInstructionKey(method.Type.Name,
			method.Name, method.Parameters.Count);
		if (compiledMethodKeys.Add(methodKey))
			methodsToCompile.Enqueue(method);
	}

	public void Dispose() => package.Dispose();
}