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
		BuildFunction(methodName, [], instructions, Platform.Linux);

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
		module += "\n" + BuildFunction(methodName, [], [.. instructions], platform, methodInfos);
		foreach (var methodInfo in methodInfos.Values)
			module += "\n" + BuildFunction(methodInfo.Symbol, methodInfo.ParameterNames,
				methodInfo.Instructions, platform, methodInfos);
		module += "\n" + BuildEntryPoint(methodName);
		if (platform == Platform.Windows && hasNumericPrint)
			module += "\n" + BuildWindowsPrintNumberHelper();
		var stringConstants = CollectPrintStrings([.. instructions], platform);
		foreach (var methodInfo in methodInfos.Values)
		foreach (var (label, text) in CollectPrintStrings(methodInfo.Instructions, platform))
			//ncrunch: no coverage start
			if (stringConstants.All(existing => existing.Label != label))
				stringConstants.Add((label, text));
		//ncrunch: no coverage end
		if (stringConstants.Count > 0)
			module += "\n" + BuildStringConstants(stringConstants);
		return module;
	}

	public bool IsPlatformUsingStdLibAndHasPrintInstructions(Platform platform,
		List<Instruction> optimizedInstructions,
		IReadOnlyDictionary<string, List<Instruction>>? precompiledMethods) =>
		platform == Platform.Linux && (HasPrintInstructions(optimizedInstructions) || //ncrunch: no coverage
			(precompiledMethods?.Values.Any(HasPrintInstructions) ?? false));

	public static bool HasPrintInstructions(IList<Instruction> instructions) =>
		instructions.OfType<PrintInstruction>().Any();

	private static string BuildModuleHeader(Platform platform, bool hasPrint, bool hasNumericPrint)
	{
		var targetTriple = platform switch
		{
			Platform.Windows => "x86_64-pc-windows-msvc",
			Platform.Linux => "x86_64-unknown-linux-gnu",
			Platform.MacOS => "x86_64-apple-macosx",
			_ => throw new NotSupportedException("Unsupported platform: " + platform) //ncrunch: no coverage
		};
		var header = $"target triple = \"{targetTriple}\"\n";
		if (platform == Platform.Windows)
			header += "@_fltused = global i32 0\n";
		if (hasPrint)
			if (platform == Platform.Windows)
			{
				header += "\ndeclare ptr @GetStdHandle(i32)\n";
				header += "declare i32 @WriteFile(ptr, ptr, i32, ptr, ptr)\n";
			}
			else
			{
				header += "\ndeclare i32 @printf(ptr, ...)\n";
				if (hasNumericPrint)
				{
					header += "declare i32 @snprintf(ptr, i64, ptr, ...)\n";
					header += "@str.safe_s = private unnamed_addr constant [3 x i8] c\"%s\\00\"\n";
				}
			}
		return header;
	}

	private static string BuildFunction(string methodName, IEnumerable<string> paramNames,
		List<Instruction> instructions, Platform platform,
		Dictionary<string, CompiledMethodInfo>? compiledMethods = null)
	{
		var parameterList = paramNames.ToList();
		var paramIndexByName = parameterList.Select((name, index) => (name, index)).
			ToDictionary(x => x.name, x => x.index);
		var paramSignature =
			string.Join(", ", parameterList.Select((_, index) => $"double %param{index}"));
		var lines = new List<string>
		{
			$"define double @{methodName}({paramSignature}) {{",
			"entry:"
		};
		var context = new EmitContext(paramIndexByName, instructions, compiledMethods, platform);
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
			lines.Add("  ret double 0.0"); //ncrunch: no coverage
		lines.Add("}");
		return string.Join("\n", lines);
	}

	private sealed class EmitContext(Dictionary<string, int> paramIndexByName,
		List<Instruction> instructions, Dictionary<string, CompiledMethodInfo>? compiledMethods,
		Platform platform)
	{
		public Dictionary<string, int> ParamIndexByName { get; } = paramIndexByName;
		public Dictionary<string, CompiledMethodInfo>? CompiledMethods { get; } = compiledMethods;
		public Platform Platform { get; } = platform;
		public Dictionary<Register, List<Expression>> RegisterInstances { get; } = new();
		public Dictionary<string, List<Expression>> VariableInstances { get; } = new(StringComparer.Ordinal);
		public Dictionary<int, string> BlockLabels { get; } = BuildBlockLabels(instructions);
		public Dictionary<int, int> JumpEndPositions { get; } = BuildJumpEndPositions(instructions);
		public Dictionary<Register, string> RegisterValues { get; } = new();
		public Dictionary<string, string> VariablePointers { get; } = new(StringComparer.Ordinal);
		public HashSet<string> AllocatedVariables { get; } = new(StringComparer.Ordinal);
		public HashSet<string> TerminatedBlocks { get; } = new(StringComparer.Ordinal);
		public string CurrentBlock { get; set; } = "entry";
		public string? LastConditionTemp { get; set; }
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
				//ncrunch: no coverage start
				AddLabel(labels, index + jumpIf.Steps + 1, ref labelIndex);
				break;
			case JumpToId { InstructionType: InstructionType.JumpEnd }:
				AddLabel(labels, index, ref labelIndex);
				break;
			case JumpToId:
				AddLabel(labels, index + 1, ref labelIndex);
				break; //ncrunch: no coverage end
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
				positions[jumpEnd.Id] = index; //ncrunch: no coverage
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
			break; //ncrunch: no coverage
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
			break; //ncrunch: no coverage
		case JumpToId jumpToId:
			EmitJumpToId(jumpToId, lines, context, index); //ncrunch: no coverage
			break; //ncrunch: no coverage
		default:
			throw new NotSupportedException(
				$"LLVM IR compilation does not support instruction: {instruction.GetType().Name} ({instruction.InstructionType})");
		}
	}

	private static void EmitStoreVariable(StoreVariableInstruction store, List<string> lines,
		EmitContext context)
	{
		if (store.ValueInstance.IsText)
			return;
		//ncrunch: no coverage start
		EnsureVariableAllocated(store.Identifier, lines, context);
		var value = FormatDouble(store.ValueInstance.Number);
		lines.Add($"  store double {value}, ptr {context.VariablePointers[store.Identifier]}");
	} //ncrunch: no coverage end

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
		return 1; //ncrunch: no coverage
	}

	private static void EmitLoadVariable(LoadVariableToRegister loadVar, List<string> lines,
		EmitContext context)
	{
		if (context.VariableInstances.TryGetValue(loadVar.Identifier, out var instances))
		{ //ncrunch: no coverage start
			context.RegisterInstances[loadVar.Register] = instances;
			return;
		} //ncrunch: no coverage end
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
		if (!loadConst.ValueInstance.IsText)
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
			_ => throw new NotSupportedException( //ncrunch: no coverage
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
			_ => throw new NotSupportedException( //ncrunch: no coverage
				$"LLVM IR comparison {binary.InstructionType} is not supported")
		};
		var temp = context.NextTemp();
		lines.Add($"  {temp} = fcmp {predicate} double {left}, {right}");
		context.LastConditionTemp = temp;
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
		var printKey = BuildPrintKey(print, context.Platform);
		var stringLabel = "@str." + BuildPrintLabel(printKey);
		var strGep = context.NextTemp();
		lines.Add(
			$"  {strGep} = getelementptr inbounds [0 x i8], ptr {stringLabel}, i64 0, i64 0");
		if (context.Platform == Platform.Windows)
		{
			var stdoutHandle = context.NextTemp();
			lines.Add($"  {stdoutHandle} = call ptr @GetStdHandle(i32 -11)");
			if (print.ValueRegister.HasValue && !print.ValueIsText)
			{
				var prefixLength = System.Text.Encoding.UTF8.GetByteCount(print.TextPrefix);
				if (prefixLength > 0)
				{
					var writtenPrefix = context.NextTemp();
					lines.Add($"  {writtenPrefix} = alloca i32");
					lines.Add(
						$"  call i32 @WriteFile(ptr {stdoutHandle}, ptr {strGep}, i32 {prefixLength}, ptr {writtenPrefix}, ptr null)");
				}
				var numValue = GetRegisterValue(print.ValueRegister.Value, context);
				lines.Add(
					$"  call void @print_number_from_double(ptr {stdoutHandle}, double {numValue})");
			}
			else
			{ //ncrunch: no coverage start
				var textLength = System.Text.Encoding.UTF8.GetByteCount(print.TextPrefix) + 1;
				var writtenText = context.NextTemp();
				lines.Add($"  {writtenText} = alloca i32");
				lines.Add(
					$"  call i32 @WriteFile(ptr {stdoutHandle}, ptr {strGep}, i32 {textLength}, ptr {writtenText}, ptr null)");
			} //ncrunch: no coverage end
			return;
		}
		if (print.ValueRegister.HasValue && !print.ValueIsText)
		{
			var numValue = GetRegisterValue(print.ValueRegister.Value, context);
			var bufPtr = context.NextTemp();
			lines.Add($"  {bufPtr} = alloca [64 x i8]");
			var castPtr = context.NextTemp();
			lines.Add($"  {castPtr} = getelementptr [64 x i8], ptr {bufPtr}, i64 0, i64 0");
			var snprintfResult = context.NextTemp();
			lines.Add(
				$"  {snprintfResult} = call i32 (ptr, i64, ptr, ...) @snprintf(ptr {castPtr}, i64 64, ptr {strGep}, double {numValue})");
			var safeFmt = context.NextTemp();
			lines.Add(
				$"  {safeFmt} = call i32 (ptr, ...) @printf(ptr @str.safe_s, ptr {castPtr})");
		}
		else
		{
			var result = context.NextTemp();
			lines.Add($"  {result} = call i32 (ptr, ...) @printf(ptr {strGep})");
		}
	}

	private static string BuildPrintKey(PrintInstruction print, Platform platform) =>
		platform == Platform.Windows && print.ValueRegister.HasValue && !print.ValueIsText
			? print.TextPrefix
			: print.ValueRegister.HasValue && !print.ValueIsText
				? print.TextPrefix + "%g"
				: print.TextPrefix;

	private static List<(string Label, string Text)> CollectPrintStrings(
		List<Instruction> instructions, Platform platform)
	{
		var strings = new List<(string, string)>();
		var seen = new HashSet<string>(StringComparer.Ordinal);
		foreach (var print in instructions.OfType<PrintInstruction>())
		{
			var key = BuildPrintKey(print, platform);
			if (seen.Add(key))
				strings.Add(("str." + BuildPrintLabel(key), key + "\n\0"));
		}
		return strings;
	}

	private static string BuildWindowsPrintNumberHelper() =>
		string.Join("\n", "define void @print_number_from_double(ptr %stdout, double %value) {",
			"entry:",
			"  %buffer = alloca [64 x i8]",
			"  %bufferStart = getelementptr [64 x i8], ptr %buffer, i64 0, i64 0",
			"  %remainingPtr = alloca i64",
			"  %writeIndexPtr = alloca i64",
			"  %writtenPtr = alloca i32",
			"  %number = fptosi double %value to i64",
			"  %isNegative = icmp slt i64 %number, 0",
			"  %negated = sub i64 0, %number",
			"  %absolute = select i1 %isNegative, i64 %negated, i64 %number",
			"  store i64 %absolute, ptr %remainingPtr",
			"  store i64 62, ptr %writeIndexPtr",
			"  %newlinePtr = getelementptr i8, ptr %bufferStart, i64 62",
			"  store i8 10, ptr %newlinePtr",
			"  %isZero = icmp eq i64 %absolute, 0",
			"  br i1 %isZero, label %storeZero, label %digitLoop",
			"storeZero:",
			"  %zeroIndex = load i64, ptr %writeIndexPtr",
			"  %zeroStoreIndex = sub i64 %zeroIndex, 1",
			"  store i64 %zeroStoreIndex, ptr %writeIndexPtr",
			"  %zeroPtr = getelementptr i8, ptr %bufferStart, i64 %zeroStoreIndex",
			"  store i8 48, ptr %zeroPtr",
			"  br label %afterDigits",
			"digitLoop:",
			"  %current = load i64, ptr %remainingPtr",
			"  %remainder = urem i64 %current, 10",
			"  %quotient = udiv i64 %current, 10",
			"  store i64 %quotient, ptr %remainingPtr",
			"  %digitValue = add i64 %remainder, 48",
			"  %digitByte = trunc i64 %digitValue to i8",
			"  %loopIndex = load i64, ptr %writeIndexPtr",
			"  %digitStoreIndex = sub i64 %loopIndex, 1",
			"  store i64 %digitStoreIndex, ptr %writeIndexPtr",
			"  %digitPtr = getelementptr i8, ptr %bufferStart, i64 %digitStoreIndex",
			"  store i8 %digitByte, ptr %digitPtr",
			"  %hasMoreDigits = icmp ne i64 %quotient, 0",
			"  br i1 %hasMoreDigits, label %digitLoop, label %afterDigits",
			"afterDigits:",
			"  br i1 %isNegative, label %storeSign, label %prepareWrite",
			"storeSign:",
			"  %signIndex = load i64, ptr %writeIndexPtr",
			"  %signStoreIndex = sub i64 %signIndex, 1",
			"  store i64 %signStoreIndex, ptr %writeIndexPtr",
			"  %signPtr = getelementptr i8, ptr %bufferStart, i64 %signStoreIndex",
			"  store i8 45, ptr %signPtr",
			"  br label %prepareWrite",
			"prepareWrite:",
			"  %startIndex = load i64, ptr %writeIndexPtr",
			"  %outputPtr = getelementptr i8, ptr %bufferStart, i64 %startIndex",
			"  %length64 = sub i64 63, %startIndex",
			"  %length32 = trunc i64 %length64 to i32",
			"  call i32 @WriteFile(ptr %stdout, ptr %outputPtr, i32 %length32, ptr %writtenPtr, ptr null)",
			"  ret void",
			"}");

	private static void EmitJump(Jump jump, List<string> lines, EmitContext context, int index)
	{
		var target = index + jump.InstructionsToSkip + 1;
		if (context.BlockLabels.TryGetValue(target, out var label))
			switch (jump.InstructionType)
			{
			case InstructionType.JumpIfTrue or InstructionType.JumpIfFalse:
				var condition = context.LastConditionTemp ?? "%t0";
				var fallthrough = context.NextTemp();
				var fallthroughLabel = $"fall{fallthrough[1..]}";
				context.BlockLabels[index + 1] = fallthroughLabel;
				lines.Add(jump.InstructionType == InstructionType.JumpIfFalse
					? $"  br i1 {condition}, label %{fallthroughLabel}, label %{label}"
					: $"  br i1 {condition}, label %{label}, label %{fallthroughLabel}");
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

	//ncrunch: no coverage start
	private static void EmitJumpToId(JumpToId jumpToId, List<string> lines, EmitContext context,
		int index)
	{
		if (!context.JumpEndPositions.TryGetValue(jumpToId.Id, out var endIndex) ||
			!context.BlockLabels.TryGetValue(endIndex, out var label))
			return;
		switch (jumpToId.InstructionType)
		{
		case InstructionType.JumpToIdIfFalse or InstructionType.JumpToIdIfTrue:
			var condition = context.LastConditionTemp ?? "%t0";
			var fallthroughLabel = context.BlockLabels.TryGetValue(index + 1, out var existing)
				? existing
				: $"fallid{context.TempCounter++}";
			if (!context.BlockLabels.ContainsKey(index + 1))
				context.BlockLabels[index + 1] = fallthroughLabel;
			lines.Add(jumpToId.InstructionType == InstructionType.JumpToIdIfFalse
				? $"  br i1 {condition}, label %{fallthroughLabel}, label %{label}"
				: $"  br i1 {condition}, label %{label}, label %{fallthroughLabel}");
			context.TerminatedBlocks.Add(context.CurrentBlock);
			lines.Add($"{fallthroughLabel}:");
			context.CurrentBlock = fallthroughLabel;
			break;
		default:
			lines.Add($"  br label %{label}");
			context.TerminatedBlocks.Add(context.CurrentBlock);
			break;
		}
	} //ncrunch: no coverage end

	private static void EmitInvoke(Invoke invoke, List<string> lines, EmitContext context)
	{
		if (invoke.Method == null)
			throw new NotSupportedException("Invoke instruction is missing method metadata"); //ncrunch: no coverage
		if (invoke.Method.Method.Name == Method.From && invoke.Method.Instance == null)
		{
			context.RegisterInstances[invoke.Register] = ResolveConstructorArguments(invoke.Method);
			return;
		}
		var methodKey = BytecodeDeserializer.BuildMethodInstructionKey(invoke.Method.Method.Type.Name,
			invoke.Method.Method.Name, invoke.Method.Method.Parameters.Count);
		if (context.CompiledMethods == null ||
			!context.CompiledMethods.TryGetValue(methodKey, out var methodInfo))
			throw new NotSupportedException( //ncrunch: no coverage
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
		//ncrunch: no coverage start
		var variableName = expression.ToString();
		if (context.ParamIndexByName.TryGetValue(variableName, out var paramIndex))
			return $"%param{paramIndex}";
		if (context.VariablePointers.ContainsKey(variableName))
			throw new NotSupportedException(
				"Cannot pass stack variable as inline argument in LLVM IR: " + variableName);
		throw new NotSupportedException(
			"Unsupported expression for LLVM IR native compilation: " + expression);
	} //ncrunch: no coverage end

	private static List<Expression> ResolveConstructorArguments(MethodCall constructorCall)
	{
		var members = constructorCall.ReturnType.Members.Where(member => !member.Type.IsTrait).ToList();
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
			return ResolveConstructorArguments(constructorCall); //ncrunch: no coverage
		var instanceName = methodCall.Instance?.ToString();
		if (instanceName != null && variableInstances.TryGetValue(instanceName, out var values))
			return values;
		throw new NotSupportedException( //ncrunch: no coverage
			"Cannot resolve instance values for method call: " + methodCall);
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
			{ //ncrunch: no coverage start
				if (includeMembers && existing.MemberNames.Count == 0)
					methods[methodKey] = BuildMethodInfo(method, true, precompiledMethods);
				continue;
			} //ncrunch: no coverage end
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
			? method.Type.Members.Where(member => !member.Type.IsTrait).Select(member => member.Name).ToList()
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

	//ncrunch: no coverage start
	private static List<Instruction> GenerateInstructions(Method method)
	{
		var body = method.GetBodyAndParseIfNeeded();
		var expressions = body is Body b
			? b.Expressions
			: [body];
		var arguments =
			method.Parameters.ToDictionary(parameter => parameter.Name, //ncrunch: no coverage
				parameter => new ValueInstance(parameter.Type, 0)); //ncrunch: no coverage
		return new BytecodeGenerator(new InvokedMethod(expressions, arguments, method.ReturnType),
			new Registry()).Generate();
	} //ncrunch: no coverage end

	private static string BuildEntryPoint(string methodName) =>
		string.Join("\n",
			new[]
			{
				"define i32 @main() {",
				"entry:",
				$"  %result = call double @{methodName}()",
				"  ret i32 0", "}"
			});

	private static string BuildPrintLabel(string text)
	{
		var result = new System.Text.StringBuilder(text.Length * 2);
		foreach (var character in text)
			if (char.IsLetterOrDigit(character))
				result.Append(character);
			else
				result.Append('_').Append(((int)character).ToString("X4"));
		return result.ToString();
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
		foreach (var character in text)
			if (character == '\n')
				result.Append("\\0A");
			else if (character == '\0')
				result.Append("\\00");
			else if (character == '\\')
				result.Append("\\5C"); //ncrunch: no coverage
			else if (character == '"')
				result.Append("\\22"); //ncrunch: no coverage
			else if (character is >= ' ' and <= '~')
				result.Append(character);
			else
				result.Append($"\\{(int)character:X2}"); //ncrunch: no coverage
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

	private static string GetRegisterValue(Register register, EmitContext context) =>
		context.RegisterValues.GetValueOrDefault(register, "0.0");

	private static bool HasNumericPrint(IEnumerable<Instruction> instructions) =>
		instructions.OfType<PrintInstruction>().Any(print => print.ValueRegister.HasValue && !print.ValueIsText);

	private static string FormatDouble(double value) =>
		value == 0.0
			? "0.0"
			: value == (long)value
				? $"{value:F1}"
				: value.ToString("G17", System.Globalization.CultureInfo.InvariantCulture);
}
