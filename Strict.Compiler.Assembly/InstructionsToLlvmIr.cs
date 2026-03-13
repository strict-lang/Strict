using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;

namespace Strict.Compiler.Assembly;

/// <summary>
/// Compiles Strict bytecode instructions to LLVM IR text. Generates typed SSA form IR that can be
/// compiled directly by clang or llc, benefiting from LLVM's optimization passes (-O2 by default)
/// and platform-specific code generation. Much simpler than raw NASM: no manual register allocation,
/// no ABI handling, no stack frame management — LLVM handles all of this.
/// </summary>
public sealed class InstructionsToLlvmIr : InstructionsCompiler
{
	private sealed class CompiledMethodInfo(string symbol,
		List<Instruction> instructions, List<string> parameterNames, List<string> memberNames)
	{
		public string Symbol { get; } = symbol;
		public List<Instruction> Instructions { get; } = instructions;
		public List<string> ParameterNames { get; } = parameterNames;
		public List<string> MemberNames { get; } = memberNames;
	}

	/// <summary>
	/// Compiles a single method's instructions into an LLVM IR function (no entry point).
	/// </summary>
	public string CompileInstructions(string methodName, List<Instruction> instructions) =>
		BuildFunction(methodName, [], instructions);

	/// <summary>
	/// Produces a complete LLVM IR module for the target platform including the compiled function,
	/// any called methods, and a main/entry point that calls the function and exits.
	/// </summary>
	public string CompileForPlatform(string methodName, IList<Instruction> instructions,
		Platform platform, IReadOnlyDictionary<string, List<Instruction>>? precompiledMethods = null)
	{
		var hasPrint = instructions.OfType<PrintInstruction>().Any();
		var methodInfos = CollectMethods([.. instructions], precompiledMethods);
		var hasNumericPrint = HasNumericPrint(instructions) ||
			methodInfos.Values.Any(info => HasNumericPrint(info.Instructions));
		var module = BuildModuleHeader(platform, hasPrint, hasNumericPrint);
		module += "\n" + BuildFunction(methodName, [], [.. instructions], methodInfos);
		foreach (var methodInfo in methodInfos.Values)
			module += "\n" + BuildFunction(methodInfo.Symbol, methodInfo.ParameterNames,
				methodInfo.Instructions, methodInfos);
		module += "\n" + BuildEntryPoint(methodName, platform, hasPrint);
		var stringConstants = CollectPrintStrings([.. instructions]);
		foreach (var methodInfo in methodInfos.Values)
			foreach (var (label, text) in CollectPrintStrings(methodInfo.Instructions))
				if (stringConstants.All(existing => existing.Label != label))
					stringConstants.Add((label, text));
		if (stringConstants.Count > 0)
			module += "\n" + BuildStringConstants(stringConstants);
		return module;
	}

	public bool HasPrintInstructions(IList<Instruction> instructions) =>
		instructions.OfType<PrintInstruction>().Any();

	private static string BuildModuleHeader(Platform platform, bool hasPrint, bool hasNumericPrint)
	{
		var targetTriple = platform switch
		{
			Platform.Windows => "x86_64-pc-windows-msvc",
			Platform.Linux => "x86_64-unknown-linux-gnu",
			Platform.MacOS => "x86_64-apple-macosx",
			_ => throw new NotSupportedException("Unsupported platform: " + platform)
		};
		var header = $"target triple = \"{targetTriple}\"\n";
		if (hasPrint)
		{
			header += "\ndeclare i32 @printf(ptr, ...)\n";
			if (hasNumericPrint)
				header += "declare i32 @snprintf(ptr, i64, ptr, ...)\n";
		}
		return header;
	}

	private static string BuildFunction(string methodName, IEnumerable<string> paramNames,
		List<Instruction> instructions,
		Dictionary<string, CompiledMethodInfo>? compiledMethods = null)
	{
		var parameterList = paramNames.ToList();
		var paramIndexByName = parameterList.Select((name, index) => (name, index))
			.ToDictionary(x => x.name, x => x.index);
		var paramSignature = string.Join(", ",
			parameterList.Select((name, index) => $"double %param{index}"));
		var lines = new List<string>
		{
			$"define double @{methodName}({paramSignature}) {{",
			"entry:"
		};
		var context = new EmitContext(paramIndexByName, instructions, compiledMethods);
		for (var index = 0; index < instructions.Count; index++)
		{
			if (context.BlockLabels.TryGetValue(index, out var label))
			{
				if (!context.TerminatedBlocks.Contains(context.CurrentBlock))
					lines.Add($"  br label %{label}");
				lines.Add($"{label}:");
				context.CurrentBlock = label;
			}
			EmitInstruction(instructions[index], lines, context, index);
		}
		if (!context.HasReturn)
			lines.Add("  ret double 0.0");
		lines.Add("}");
		return string.Join("\n", lines);
	}

	private sealed class EmitContext
	{
		public EmitContext(Dictionary<string, int> paramIndexByName,
			List<Instruction> instructions,
			Dictionary<string, CompiledMethodInfo>? compiledMethods)
		{
			ParamIndexByName = paramIndexByName;
			CompiledMethods = compiledMethods;
			RegisterInstances = new Dictionary<Register, List<Expression>>();
			VariableInstances = new Dictionary<string, List<Expression>>(StringComparer.Ordinal);
			BlockLabels = BuildBlockLabels(instructions);
			JumpEndPositions = BuildJumpEndPositions(instructions);
		}

		public Dictionary<string, int> ParamIndexByName { get; }
		public Dictionary<string, CompiledMethodInfo>? CompiledMethods { get; }
		public Dictionary<Register, List<Expression>> RegisterInstances { get; }
		public Dictionary<string, List<Expression>> VariableInstances { get; }
		public Dictionary<int, string> BlockLabels { get; }
		public Dictionary<int, int> JumpEndPositions { get; }
		public Dictionary<Register, string> RegisterValues { get; } = new();
		public Dictionary<string, string> VariablePointers { get; } = new(StringComparer.Ordinal);
		public HashSet<string> AllocatedVariables { get; } = new(StringComparer.Ordinal);
		public HashSet<string> TerminatedBlocks { get; } = new(StringComparer.Ordinal);
		public string CurrentBlock { get; set; } = "entry";
		public int TempCounter { get; set; }
		public bool HasReturn { get; set; }
		public string NextTemp() => $"%t{TempCounter++}";
	}

	private static Dictionary<int, string> BuildBlockLabels(List<Instruction> instructions)
	{
		var labels = new Dictionary<int, string>();
		var labelIndex = 0;
		for (var index = 0; index < instructions.Count; index++)
			switch (instructions[index])
			{
			case Jump jump:
				AddLabel(labels, index + jump.InstructionsToSkip + 1, ref labelIndex);
				break;
			case JumpIf jumpIf:
				AddLabel(labels, index + jumpIf.Steps + 1, ref labelIndex);
				break;
			case JumpToId { InstructionType: InstructionType.JumpEnd } jumpEnd:
				AddLabel(labels, index, ref labelIndex);
				break;
			case JumpToId jumpToId:
				AddLabel(labels, index + 1, ref labelIndex);
				break;
			}
		return labels;
	}

	private static void AddLabel(Dictionary<int, string> labels, int target, ref int labelIndex)
	{
		if (target >= 0 && !labels.ContainsKey(target))
			labels[target] = $"L{labelIndex++}";
	}

	private static Dictionary<int, int> BuildJumpEndPositions(List<Instruction> instructions)
	{
		var positions = new Dictionary<int, int>();
		for (var index = 0; index < instructions.Count; index++)
			if (instructions[index] is JumpToId { InstructionType: InstructionType.JumpEnd } jumpEnd)
				positions[jumpEnd.Id] = index;
		return positions;
	}

	private static void EmitInstruction(Instruction instruction, List<string> lines,
		EmitContext context, int index)
	{
		switch (instruction)
		{
		case StoreVariableInstruction storeConst
			when !context.ParamIndexByName.ContainsKey(storeConst.Identifier):
			EmitStoreVariable(storeConst, lines, context);
			break;
		case StoreVariableInstruction:
			break;
		case LoadVariableToRegister loadVar:
			EmitLoadVariable(loadVar, lines, context);
			break;
		case LoadConstantInstruction loadConst:
			EmitLoadConstant(loadConst, context);
			break;
		case BinaryInstruction binary when !binary.IsConditional():
			EmitArithmetic(binary, lines, context);
			break;
		case BinaryInstruction binary:
			EmitComparison(binary, lines, context);
			break;
		case StoreFromRegisterInstruction storeReg:
			EmitStoreFromRegister(storeReg, lines, context);
			break;
		case ReturnInstruction ret:
			EmitReturn(ret, lines, context);
			break;
		case PrintInstruction print:
			EmitPrint(print, lines, context);
			break;
		case Jump jump:
			EmitJump(jump, lines, context, index);
			break;
		case Invoke invoke:
			EmitInvoke(invoke, lines, context);
			break;
		case JumpToId { InstructionType: InstructionType.JumpEnd }:
			break;
		case JumpToId jumpToId:
			EmitJumpToId(jumpToId, lines, context, index);
			break;
		}
	}

	private static void EmitStoreVariable(StoreVariableInstruction store, List<string> lines,
		EmitContext context)
	{
		if (store.ValueInstance.IsText)
			return;
		EnsureVariableAllocated(store.Identifier, lines, context);
		var value = FormatDouble(store.ValueInstance.Number);
		lines.Add($"  store double {value}, ptr {context.VariablePointers[store.Identifier]}");
	}

	private static void EnsureVariableAllocated(string name, List<string> lines,
		EmitContext context)
	{
		if (context.AllocatedVariables.Add(name))
		{
			var pointer = $"%var.{name}";
			context.VariablePointers[name] = pointer;
			lines.Insert(FindEntryInsertPoint(lines), $"  {pointer} = alloca double");
		}
	}

	private static int FindEntryInsertPoint(List<string> lines)
	{
		for (var index = 0; index < lines.Count; index++)
			if (lines[index] == "entry:")
				return index + 1;
		return 1;
	}

	private static void EmitLoadVariable(LoadVariableToRegister loadVar, List<string> lines,
		EmitContext context)
	{
		if (context.VariableInstances.TryGetValue(loadVar.Identifier, out var instances))
		{
			context.RegisterInstances[loadVar.Register] = instances;
			return;
		}
		if (context.ParamIndexByName.TryGetValue(loadVar.Identifier, out var paramIndex))
		{
			context.RegisterValues[loadVar.Register] = $"%param{paramIndex}";
			return;
		}
		if (context.VariablePointers.TryGetValue(loadVar.Identifier, out var pointer))
		{
			var temp = context.NextTemp();
			lines.Add($"  {temp} = load double, ptr {pointer}");
			context.RegisterValues[loadVar.Register] = temp;
		}
	}

	private static void EmitLoadConstant(LoadConstantInstruction loadConst, EmitContext context)
	{
		if (loadConst.ValueInstance.IsText)
			return;
		context.RegisterValues[loadConst.Register] = FormatDouble(loadConst.ValueInstance.Number);
	}

	private static void EmitArithmetic(BinaryInstruction binary, List<string> lines,
		EmitContext context)
	{
		var left = GetRegisterValue(binary.Registers[0], context);
		var right = GetRegisterValue(binary.Registers[1], context);
		var dest = context.NextTemp();
		var op = binary.InstructionType switch
		{
			InstructionType.Add => "fadd",
			InstructionType.Subtract => "fsub",
			InstructionType.Multiply => "fmul",
			InstructionType.Divide => "fdiv",
			InstructionType.Modulo => "frem",
			_ => throw new NotSupportedException(
				$"LLVM IR compilation of {binary.InstructionType} is not supported")
		};
		lines.Add($"  {dest} = {op} double {left}, {right}");
		context.RegisterValues[binary.Registers[^1]] = dest;
	}

	private static void EmitComparison(BinaryInstruction binary, List<string> lines,
		EmitContext context)
	{
		var left = GetRegisterValue(binary.Registers[0], context);
		var right = GetRegisterValue(binary.Registers[1], context);
		var predicate = binary.InstructionType switch
		{
			InstructionType.Equal => "oeq",
			InstructionType.NotEqual => "one",
			InstructionType.LessThan => "olt",
			InstructionType.GreaterThan => "ogt",
			_ => throw new NotSupportedException(
				$"LLVM IR comparison {binary.InstructionType} is not supported")
		};
		var temp = context.NextTemp();
		lines.Add($"  {temp} = fcmp {predicate} double {left}, {right}");
		context.RegisterValues[binary.Registers[^1]] = temp;
	}

	private static void EmitStoreFromRegister(StoreFromRegisterInstruction storeReg,
		List<string> lines, EmitContext context)
	{
		if (context.RegisterInstances.TryGetValue(storeReg.Register, out var constructorArgs))
		{
			context.VariableInstances[storeReg.Identifier] = constructorArgs;
			return;
		}
		EnsureVariableAllocated(storeReg.Identifier, lines, context);
		var value = GetRegisterValue(storeReg.Register, context);
		lines.Add($"  store double {value}, ptr {context.VariablePointers[storeReg.Identifier]}");
	}

	private static void EmitReturn(ReturnInstruction ret, List<string> lines,
		EmitContext context)
	{
		var value = GetRegisterValue(ret.Register, context);
		lines.Add($"  ret double {value}");
		context.HasReturn = true;
		context.TerminatedBlocks.Add(context.CurrentBlock);
	}

	private static void EmitPrint(PrintInstruction print, List<string> lines,
		EmitContext context)
	{
		var printKey = BuildPrintKey(print);
		var stringLabel = "@str." + SanitizeLabel(printKey);
		if (print.ValueRegister.HasValue && !print.ValueIsText)
		{
			var numValue = GetRegisterValue(print.ValueRegister.Value, context);
			var bufPtr = context.NextTemp();
			lines.Add($"  {bufPtr} = alloca [64 x i8]");
			var castPtr = context.NextTemp();
			lines.Add($"  {castPtr} = getelementptr [64 x i8], ptr {bufPtr}, i64 0, i64 0");
			var snprintfResult = context.NextTemp();
			lines.Add(
				$"  {snprintfResult} = call i32 (ptr, i64, ptr, ...) @snprintf(ptr {castPtr}, i64 64, ptr {stringLabel}, double {numValue})");
			var printResult = context.NextTemp();
			lines.Add(
				$"  {printResult} = call i32 (ptr, ...) @printf(ptr {castPtr})");
		}
		else
		{
			var result = context.NextTemp();
			lines.Add($"  {result} = call i32 (ptr, ...) @printf(ptr {stringLabel})");
		}
	}

	private static void EmitJump(Jump jump, List<string> lines, EmitContext context, int index)
	{
		var target = index + jump.InstructionsToSkip + 1;
		if (!context.BlockLabels.TryGetValue(target, out var label))
			return;
		switch (jump.InstructionType)
		{
		case InstructionType.JumpIfTrue or InstructionType.JumpIfFalse:
			var lastComparisonValue = FindLastComparisonValue(context);
			var fallthrough = context.NextTemp();
			var fallthroughLabel = $"fall{fallthrough[1..]}";
			context.BlockLabels[index + 1] = fallthroughLabel;
			if (jump.InstructionType == InstructionType.JumpIfFalse)
				lines.Add($"  br i1 {lastComparisonValue}, label %{fallthroughLabel}, label %{label}");
			else
				lines.Add($"  br i1 {lastComparisonValue}, label %{label}, label %{fallthroughLabel}");
			context.TerminatedBlocks.Add(context.CurrentBlock);
			lines.Add($"{fallthroughLabel}:");
			context.CurrentBlock = fallthroughLabel;
			break;
		default:
			lines.Add($"  br label %{label}");
			context.TerminatedBlocks.Add(context.CurrentBlock);
			break;
		}
	}

	private static void EmitJumpToId(JumpToId jumpToId, List<string> lines,
		EmitContext context, int index)
	{
		if (!context.JumpEndPositions.TryGetValue(jumpToId.Id, out var endIndex) ||
			!context.BlockLabels.TryGetValue(endIndex, out var label))
			return;
		switch (jumpToId.InstructionType)
		{
		case InstructionType.JumpToIdIfFalse or InstructionType.JumpToIdIfTrue:
			var lastComparisonValue = FindLastComparisonValue(context);
			var fallthroughLabel = context.BlockLabels.TryGetValue(index + 1, out var existing)
				? existing
				: $"fallid{context.TempCounter++}";
			if (!context.BlockLabels.ContainsKey(index + 1))
				context.BlockLabels[index + 1] = fallthroughLabel;
			if (jumpToId.InstructionType == InstructionType.JumpToIdIfFalse)
				lines.Add($"  br i1 {lastComparisonValue}, label %{fallthroughLabel}, label %{label}");
			else
				lines.Add($"  br i1 {lastComparisonValue}, label %{label}, label %{fallthroughLabel}");
			context.TerminatedBlocks.Add(context.CurrentBlock);
			lines.Add($"{fallthroughLabel}:");
			context.CurrentBlock = fallthroughLabel;
			break;
		default:
			lines.Add($"  br label %{label}");
			context.TerminatedBlocks.Add(context.CurrentBlock);
			break;
		}
	}

	private static void EmitInvoke(Invoke invoke, List<string> lines, EmitContext context)
	{
		if (invoke.Method == null)
			throw new NotSupportedException("Invoke instruction is missing method metadata");
		if (invoke.Method.Method.Name == Method.From && invoke.Method.Instance == null)
		{
			context.RegisterInstances[invoke.Register] = ResolveConstructorArguments(invoke.Method);
			return;
		}
		var methodKey = BytecodeDeserializer.BuildMethodInstructionKey(invoke.Method.Method.Type.Name,
			invoke.Method.Method.Name, invoke.Method.Method.Parameters.Count);
		if (context.CompiledMethods == null ||
			!context.CompiledMethods.TryGetValue(methodKey, out var methodInfo))
			throw new NotSupportedException(
				"Non-print method calls cannot be compiled to LLVM IR. " +
				"Use the interpreted runner for programs with complex runtime method calls.");
		var arguments = new List<string>();
		if (methodInfo.MemberNames.Count > 0)
			foreach (var memberExpression in ResolveInstanceMemberArguments(invoke.Method,
				context.VariableInstances))
				arguments.Add("double " + ResolveExpressionValue(memberExpression, context));
		for (var argIndex = 0; argIndex < invoke.Method.Arguments.Count; argIndex++)
			arguments.Add("double " +
				ResolveExpressionValue(invoke.Method.Arguments[argIndex], context));
		var result = context.NextTemp();
		lines.Add($"  {result} = call double @{methodInfo.Symbol}({string.Join(", ", arguments)})");
		context.RegisterValues[invoke.Register] = result;
	}

	private static string ResolveExpressionValue(Expression expression, EmitContext context)
	{
		if (expression is Value value && !value.Data.IsText)
			return FormatDouble(value.Data.Number);
		var variableName = expression.ToString();
		if (context.ParamIndexByName.TryGetValue(variableName, out var paramIndex))
			return $"%param{paramIndex}";
		return "0.0";
	}

	private static List<Expression> ResolveConstructorArguments(MethodCall constructorCall)
	{
		var members = constructorCall.ReturnType.Members.Where(member => !member.Type.IsTrait)
			.ToList();
		var result = new List<Expression>(members.Count);
		for (var index = 0; index < members.Count; index++)
			result.Add(index < constructorCall.Arguments.Count
				? constructorCall.Arguments[index]
				: new Value(members[index].Type, new ValueInstance(members[index].Type, 0)));
		return result;
	}

	private static IEnumerable<Expression> ResolveInstanceMemberArguments(MethodCall methodCall,
		Dictionary<string, List<Expression>> variableInstances)
	{
		if (methodCall.Instance is MethodCall constructorCall &&
			constructorCall.Method.Name == Method.From && constructorCall.Instance == null)
			return ResolveConstructorArguments(constructorCall);
		var instanceName = methodCall.Instance?.ToString();
		if (instanceName != null && variableInstances.TryGetValue(instanceName, out var values))
			return values;
		throw new NotSupportedException(
			"Cannot resolve instance values for method call: " + methodCall);
	}

	private static string FindLastComparisonValue(EmitContext context)
	{
		var lastRegister = context.RegisterValues.LastOrDefault(
			pair => pair.Value.StartsWith("%", StringComparison.Ordinal));
		return lastRegister.Value ?? "%t0";
	}

	private static Dictionary<string, CompiledMethodInfo> CollectMethods(
		List<Instruction> instructions,
		IReadOnlyDictionary<string, List<Instruction>>? precompiledMethods)
	{
		var methods = new Dictionary<string, CompiledMethodInfo>(StringComparer.Ordinal);
		var queue = new Queue<(Method Method, bool IncludeMembers)>();
		EnqueueInvokedMethods(instructions, queue);
		while (queue.Count > 0)
		{
			var (method, includeMembers) = queue.Dequeue();
			var methodKey = BytecodeDeserializer.BuildMethodInstructionKey(method.Type.Name,
				method.Name, method.Parameters.Count);
			if (methods.TryGetValue(methodKey, out var existing))
			{
				if (includeMembers && existing.MemberNames.Count == 0)
					methods[methodKey] = BuildMethodInfo(method, true, precompiledMethods);
				continue;
			}
			var methodInfo = BuildMethodInfo(method, includeMembers, precompiledMethods);
			methods[methodKey] = methodInfo;
			EnqueueInvokedMethods(methodInfo.Instructions, queue);
		}
		return methods;
	}

	private static CompiledMethodInfo BuildMethodInfo(Method method, bool includeMembers,
		IReadOnlyDictionary<string, List<Instruction>>? precompiledMethods)
	{
		var methodKey = BytecodeDeserializer.BuildMethodInstructionKey(method.Type.Name,
			method.Name, method.Parameters.Count);
		var instructions =
			precompiledMethods != null && precompiledMethods.TryGetValue(methodKey, out var pre)
				? [.. pre]
				: GenerateInstructions(method);
		var memberNames = includeMembers
			? method.Type.Members.Where(member => !member.Type.IsTrait).Select(member => member.Name)
				.ToList()
			: new List<string>();
		var parameterNames = new List<string>(memberNames);
		parameterNames.AddRange(method.Parameters.Select(parameter => parameter.Name));
		return new CompiledMethodInfo(BuildMethodSymbol(method), instructions, parameterNames,
			memberNames);
	}

	private static string BuildMethodSymbol(Method method) =>
		method.Type.Name + "_" + method.Name + "_" + method.Parameters.Count;

	private static void EnqueueInvokedMethods(IEnumerable<Instruction> instructions,
		Queue<(Method Method, bool IncludeMembers)> queue)
	{
		foreach (var invoke in instructions.OfType<Invoke>())
			if (invoke.Method != null && invoke.Method.Method.Name != Method.From)
				queue.Enqueue((invoke.Method.Method, invoke.Method.Instance != null));
	}

	private static List<Instruction> GenerateInstructions(Method method)
	{
		var body = method.GetBodyAndParseIfNeeded();
		var expressions = body is Body b
			? b.Expressions
			: [body];
		var arguments =
			method.Parameters.ToDictionary(p => p.Name, p => new ValueInstance(p.Type, 0));
		return new BytecodeGenerator(new InvokedMethod(expressions, arguments, method.ReturnType),
			new Registry()).Generate();
	}

	private static string BuildEntryPoint(string methodName, Platform platform, bool hasPrint)
	{
		var lines = new List<string> { "define i32 @main() {", "entry:" };
		lines.Add($"  %result = call double @{methodName}()");
		if (platform == Platform.Linux || platform == Platform.MacOS)
			lines.Add("  ret i32 0");
		else
			lines.Add("  ret i32 0");
		lines.Add("}");
		return string.Join("\n", lines);
	}

	private static string BuildPrintKey(PrintInstruction print) =>
		print.ValueRegister.HasValue && !print.ValueIsText
			? print.TextPrefix + "%g"
			: print.TextPrefix;

	private static List<(string Label, string Text)> CollectPrintStrings(
		List<Instruction> instructions)
	{
		var strings = new List<(string, string)>();
		var seen = new HashSet<string>(StringComparer.Ordinal);
		foreach (var print in instructions.OfType<PrintInstruction>())
		{
			var key = BuildPrintKey(print);
			if (seen.Add(key))
				strings.Add(("str." + SanitizeLabel(key), key + "\n"));
		}
		return strings;
	}

	private static string BuildStringConstants(List<(string Label, string Text)> strings)
	{
		var lines = new List<string>();
		foreach (var (label, text) in strings)
		{
			var escaped = EscapeForLlvm(text);
			var length = CountLlvmStringBytes(escaped);
			lines.Add($"@{label} = private unnamed_addr constant [{length} x i8] c\"{escaped}\"");
		}
		return string.Join("\n", lines);
	}

	private static string EscapeForLlvm(string text)
	{
		var result = new System.Text.StringBuilder();
		foreach (var c in text)
		{
			if (c == '\n')
				result.Append("\\0A");
			else if (c == '\0')
				result.Append("\\00");
			else if (c == '\\')
				result.Append("\\5C");
			else if (c == '"')
				result.Append("\\22");
			else if (c is >= ' ' and <= '~')
				result.Append(c);
			else
				result.Append($"\\{(int)c:X2}");
		}
		return result.ToString();
	}

	private static int CountLlvmStringBytes(string escaped)
	{
		var count = 0;
		for (var index = 0; index < escaped.Length; index++)
		{
			count++;
			if (escaped[index] == '\\' && index + 2 < escaped.Length)
				index += 2;
		}
		return count;
	}

	private static string SanitizeLabel(string text) =>
		new(text.Select(c => char.IsLetterOrDigit(c) ? c : '_').ToArray());

	private static string GetRegisterValue(Register register, EmitContext context) =>
		context.RegisterValues.TryGetValue(register, out var value)
			? value
			: "0.0";

	private static bool HasNumericPrint(IEnumerable<Instruction> instructions) =>
		instructions.OfType<PrintInstruction>()
			.Any(print => print.ValueRegister.HasValue && !print.ValueIsText);

	private static string FormatDouble(double value) =>
		value == 0.0
			? "0.0"
			: value == (long)value
				? $"{value:F1}"
				: value.ToString("G17", System.Globalization.CultureInfo.InvariantCulture);
}
