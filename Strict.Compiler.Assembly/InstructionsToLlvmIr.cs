using Strict.Bytecode;
using Strict.Bytecode.Instructions;
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
	public override Task<string> Compile(BinaryExecutable binary, Platform platform)
	{
		var precompiledMethods = BuildPrecompiledMethodsInternal(binary);
		var output = CompileForPlatform(Method.Run, binary.EntryPoint.instructions, platform,
			precompiledMethods);
		return Task.FromResult(output);
	}

	public override string Extension => ".ll";

	public string CompileInstructions(string methodName, List<Instruction> instructions) =>
		BuildFunction(methodName, [], instructions, Platform.Linux);

	private static string CompileForPlatform(string methodName, IReadOnlyList<Instruction> instructions,
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
		public Dictionary<Register, Register[]> RegisterInstances { get; } = new();
		public Dictionary<string, Register[]> VariableInstances { get; } = new(StringComparer.Ordinal);
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
			case JumpToId { InstructionType: InstructionType.JumpEnd }:
				AddLabel(labels, index, ref labelIndex);
				break;
			case JumpToId:
				AddLabel(labels, index + 1, ref labelIndex);
				break;
			case Jump jump:
				AddLabel(labels, index + jump.InstructionsToSkip + 1, ref labelIndex);
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
			if (instructions[index].InstructionType == InstructionType.JumpEnd)
				positions[((JumpToId)instructions[index]).Id] = index; //ncrunch: no coverage
		return positions;
	}

	private static void EmitInstruction(Instruction instruction, List<string> lines,
		EmitContext context, int index)
	{
		switch (instruction.InstructionType)
		{
		case InstructionType.StoreConstantToVariable:
			var storeConst = (StoreVariableInstruction)instruction;
			if (!context.ParamIndexByName.ContainsKey(storeConst.Identifier))
				EmitStoreVariable(storeConst, lines, context);
			break; //ncrunch: no coverage
		case InstructionType.LoadVariableToRegister:
			var loadVar = (LoadVariableToRegister)instruction;
			EmitLoadVariable(loadVar, lines, context);
			break;
		case InstructionType.LoadConstantToRegister:
			var loadConst = (LoadConstantInstruction)instruction;
			EmitLoadConstant(loadConst, context);
			break;
		case InstructionType.Add:
		case InstructionType.Subtract:
		case InstructionType.Multiply:
		case InstructionType.Divide:
		case InstructionType.Modulo:
			EmitArithmetic((BinaryInstruction)instruction, lines, context);
			break;
		case InstructionType.Equal:
		case InstructionType.NotEqual:
		case InstructionType.LessThan:
		case InstructionType.GreaterThan:
			EmitComparison((BinaryInstruction)instruction, lines, context);
			break;
		case InstructionType.StoreRegisterToVariable:
			var storeReg = (StoreFromRegisterInstruction)instruction;
			EmitStoreFromRegister(storeReg, lines, context);
			break;
		case InstructionType.Return:
			var ret = (ReturnInstruction)instruction;
			EmitReturn(ret, lines, context);
			break;
		case InstructionType.Print:
			var print = (PrintInstruction)instruction;
			EmitPrint(print, lines, context);
			break;
		case InstructionType.Jump:
		case InstructionType.JumpIfTrue:
		case InstructionType.JumpIfFalse:
			var jump = (Jump)instruction;
			EmitJump(jump, lines, context, index);
			break;
		case InstructionType.Invoke:
			var invoke = (Invoke)instruction;
			EmitInvoke(invoke, lines, context);
			break;
		case InstructionType.JumpEnd:
			break; //ncrunch: no coverage
		case InstructionType.JumpToIdIfFalse:
		case InstructionType.JumpToIdIfTrue:
			var jumpToId = (JumpToId)instruction;
			EmitJumpToId(jumpToId, lines, context, index); //ncrunch: no coverage
			break; //ncrunch: no coverage
		default:
			throw new NotSupportedException($"LLVM IR compilation does not support instruction: {
				instruction.GetType().Name
			} ({
				instruction.InstructionType
			})");
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
		if (!loadConst.Constant.IsText)
			context.RegisterValues[loadConst.Register] = FormatDouble(loadConst.Constant.Number);
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
		if (invoke.MethodInfo == null)
			throw new NotSupportedException("Invoke instruction is missing method metadata"); //ncrunch: no coverage
		if (invoke.MethodInfo.MethodName == Method.From && !invoke.MethodInfo.InstanceRegister.HasValue)
		{
			context.RegisterInstances[invoke.Register] = invoke.MethodInfo.ArgumentRegisters;
			return;
		}
		var methodKey = BuildMethodHeaderKeyInternal(invoke.MethodInfo);
		if (context.CompiledMethods == null ||
			!context.CompiledMethods.TryGetValue(methodKey, out var methodInfo))
			throw new NotSupportedException( //ncrunch: no coverage
				"Non-print method calls cannot be compiled to LLVM IR. " +
				"Use the interpreted runner for programs with complex runtime method calls.");
		var arguments = new List<string>();
		if (methodInfo.MemberNames.Count > 0 && invoke.MethodInfo.InstanceRegister.HasValue &&
			context.RegisterInstances.TryGetValue(invoke.MethodInfo.InstanceRegister.Value,
				out var memberRegisters))
			foreach (var reg in memberRegisters)
				arguments.Add("double " + GetRegisterValue(reg, context));
		foreach (var argReg in invoke.MethodInfo.ArgumentRegisters)
			arguments.Add("double " + GetRegisterValue(argReg, context));
		var result = context.NextTemp();
		lines.Add($"  {result} = call double @{methodInfo.Symbol}({string.Join(", ", arguments)})");
		context.RegisterValues[invoke.Register] = result;
	}

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

	private static string FormatDouble(double value) =>
		value == 0.0
			? "0.0"
			: value == (long)value
				? $"{value:F1}"
				: value.ToString("G17", System.Globalization.CultureInfo.InvariantCulture);
}
