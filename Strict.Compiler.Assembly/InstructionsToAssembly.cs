using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;
using Binary = Strict.Expressions.Binary;

namespace Strict.Compiler.Assembly;

/// <summary>
/// Compiles a Strict method or pre-compiled instruction list to 64 bit NASM assembly text.
/// Strict registers R0–R15 map to XMM registers xmm0–xmm15 for numeric (double) values.
/// Follows the System V AMD64 ABI: first 8 float/double parameters in xmm0–xmm7, return in xmm0.
/// The generated NASM text can be assembled with: nasm -f win64 output.asm -o output.obj
/// </summary>
public sealed class InstructionsToAssembly : InstructionsCompiler
{
	public override Task<string> Compile(BinaryExecutable binary, Platform platform)
	{
		var precompiledMethods = BuildPrecompiledMethodsInternal(binary);
		var output = CompileForPlatform(Method.Run, binary.EntryPoint.instructions, platform,
			precompiledMethods);
		return Task.FromResult(output);
	}

	public override string Extension => ".asm";

	public string CompileInstructions(string methodName, List<Instruction> instructions) =>
		BuildAssembly(methodName, [], instructions);

	private static string CompileForPlatform(string methodName, IReadOnlyList<Instruction> instructions,
		Platform platform, IReadOnlyDictionary<string, List<Instruction>>? precompiledMethods = null)
	{
		var hasPrint = instructions.OfType<PrintInstruction>().Any();
		var methodInfos = CollectMethods([.. instructions], precompiledMethods);
		var hasNumericPrint = HasNumericPrint(instructions) ||
			methodInfos.Values.Any(methodInfo => HasNumericPrint(methodInfo.Instructions));
		var functionAsm = BuildAssembly(methodName, [], [.. instructions], platform, methodInfos);
		foreach (var methodInfo in methodInfos.Values)
			functionAsm += "\n" + BuildAssembly(methodInfo.Symbol, methodInfo.ParameterNames,
				methodInfo.Instructions, platform, methodInfos);
		if (platform == Platform.Windows && hasNumericPrint)
			functionAsm += "\n" + BuildWindowsPrintNumberHelper();
		return functionAsm + "\n" + BuildEntryPoint(methodName, platform, hasPrint);
	}

	private static string BuildEntryPoint(string methodName, Platform platform, bool hasPrint = false) =>
		platform switch
		{
			Platform.Windows => BuildWindowsEntryPoint(methodName, hasPrint),
			Platform.Linux => BuildLinuxEntryPoint(methodName, hasPrint),
			Platform.MacOS => BuildMacOsEntryPoint(methodName, hasPrint),
			_ => throw new NotSupportedException("Unsupported platform: " + platform) //ncrunch: no coverage
		};

	private static string BuildWindowsEntryPoint(string methodName, bool hasPrint) =>
		string.Join("\n", "", "extern ExitProcess", hasPrint
				? "extern GetStdHandle"
				: "", hasPrint
				? "extern WriteFile"
				: "", "", "global main", "", "main:", "    push rbp", "    mov rbp, rsp",
			"    sub rsp, 32", $"    call {methodName}", "    xor rcx, rcx", "    call ExitProcess",
			"    add rsp, 32", "    pop rbp", "    ret");

	private static string BuildLinuxEntryPoint(string methodName, bool hasPrint)
	{
		if (hasPrint)
			return string.Join("\n", "extern printf", "", "global main", "", "main:", //ncrunch: no coverage
				"    push rbp", "    mov rbp, rsp",
				$"    call {methodName}", "    mov rdi, 0", "    mov rax, 60", "    syscall");
		return string.Join("\n", "", "global _start", "", "_start:", "    push rbp", "    mov rbp, rsp",
			$"    call {methodName}", "    mov rdi, 0", "    mov rax, 60", "    syscall");
	}

	private static string BuildMacOsEntryPoint(string methodName, bool hasPrint)
	{
		var printExtern = hasPrint
			? "extern _printf\n"
			: "";
		return printExtern + string.Join("\n", "", "global _main", "", "_main:", "    push rbp", "    mov rbp, rsp",
			$"    call _{methodName}", "    xor rdi, rdi", "    mov rax, 0x2000001", "    syscall");
	}

	private static string BuildAssembly(string methodName, IEnumerable<string> paramNames,
		List<Instruction> instructions, Platform platform = Platform.Linux,
		Dictionary<string, CompiledMethodInfo>? compiledMethods = null)
	{
		var paramIndexByName = paramNames.Select((name, index) => (name, index)).
			ToDictionary(x => x.name, x => x.index);
		var variableSlots = BuildVariableSlots(paramIndexByName.Keys, instructions);
		var dataConstants = CollectConstants(instructions);
		var printStrings = CollectPrintStrings(instructions);
		var (jumpLabels, jumpEndPositions) = BuildJumpLabels(instructions);
		var optimizedReturns = new HashSet<int>();
		var lines = new List<string>();
		if (dataConstants.Count > 0 || printStrings.Count > 0)
		{
			lines.Add("section .data");
			foreach (var (label, value) in dataConstants)
				lines.Add($"    {label}: dq 0x{BitConverter.DoubleToInt64Bits(value):X16}");
			foreach (var (label, text) in printStrings)
				lines.Add($"    {label}: db {BuildStringBytes(text)}, 10, 0");
			lines.Add("");
		}
		lines.Add("section .text");
		lines.Add($"global {methodName}");
		lines.Add("");
		lines.Add($"{methodName}:");
		var frameSize = AlignTo16(variableSlots.Count * 8);
		var needsFrame = NeedsStackFrame(frameSize, instructions);
		if (needsFrame)
		{
			lines.Add("    push rbp");
			lines.Add("    mov rbp, rsp");
			if (frameSize > 0)
				lines.Add($"    sub rsp, {frameSize}");
		}
		var registerInstances = new Dictionary<Register, List<Expression>>();
		var variableInstances = new Dictionary<string, List<Expression>>(StringComparer.Ordinal);
		for (var index = 0; index < instructions.Count; index++)
		{
			if (jumpLabels.TryGetValue(index, out var label))
				lines.Add($".{label}:");
			EmitInstruction(instructions[index], lines, paramIndexByName, variableSlots, dataConstants,
				printStrings, jumpLabels, jumpEndPositions, instructions, index, platform,
				registerInstances, variableInstances, compiledMethods, optimizedReturns);
		}
		if (needsFrame)
		{
			if (frameSize > 0)
				lines.Add($"    add rsp, {frameSize}");
			lines.Add("    pop rbp");
		}
		lines.Add("    ret");
		return string.Join("\n", lines);
	}

	private static bool NeedsStackFrame(int frameSize, List<Instruction> instructions) =>
		frameSize > 0 || instructions.Any(instruction => instruction is Invoke or PrintInstruction);

	private static int AlignTo16(int size) => (size + 15) / 16 * 16;

	private static Dictionary<string, int> BuildVariableSlots(IEnumerable<string> parameterNames,
		List<Instruction> instructions)
	{
		var parameterNameSet = new HashSet<string>(parameterNames);
		var slots = new Dictionary<string, int>();
		foreach (var instruction in instructions)
		{
			var varName = instruction switch
			{
				StoreFromRegisterInstruction store => store.Identifier,
				StoreVariableInstruction store => store.Identifier,
				_ => null
			};
			if (varName != null && !parameterNameSet.Contains(varName) && !slots.ContainsKey(varName))
				slots[varName] = slots.Count;
		}
		return slots;
	}

	private static List<(string Label, double Value)> CollectConstants(
		List<Instruction> instructions)
	{
		var constants = new List<(string, double)>();
		var seenValues = new HashSet<double>();
		var index = 0;
		foreach (var instruction in instructions)
		{
			var value = instruction switch
			{
				LoadConstantInstruction load when !load.Constant.IsText => (double?)load.Constant.Number,
				StoreVariableInstruction store when !store.ValueInstance.IsText => store.ValueInstance.Number,
				_ => null
			};
			if (value is { } constantValue && constantValue != 0.0 && seenValues.Add(constantValue))
				constants.Add(($"const_{index++}", constantValue));
			if (instruction is Invoke invoke)
				foreach (var invokeValue in CollectInvokeNumericConstants(invoke.Method))
					if (invokeValue != 0.0 && seenValues.Add(invokeValue))
						constants.Add(($"const_{index++}", invokeValue));
		}
		return constants;
	}

	private static IEnumerable<double> CollectInvokeNumericConstants(MethodCall call)
	{
		foreach (var argument in call.Arguments)
		foreach (var value in CollectExpressionNumericConstants(argument))
			yield return value;
		if (call.Instance is MethodCall constructorCall)
			//ncrunch: no coverage start
			foreach (var constructorArgument in constructorCall.Arguments)
			foreach (var value in CollectExpressionNumericConstants(constructorArgument))
				yield return value;
	} //ncrunch: no coverage end

	private static IEnumerable<double> CollectExpressionNumericConstants(Expression expression)
	{
		switch (expression)
		{
		case Value { Data.IsText: false } value:
			yield return value.Data.Number;
			break;
		case Binary binary:
			//ncrunch: no coverage start
			if (binary.Instance != null)
				foreach (var nested in CollectExpressionNumericConstants(binary.Instance))
					yield return nested;
			for (var index = 0; index < binary.Arguments.Count; index++)
				foreach (var nested in CollectExpressionNumericConstants(binary.Arguments[index]))
					yield return nested;
			break;
		case MethodCall methodCall:
			foreach (var nested in CollectInvokeNumericConstants(methodCall))
				yield return nested;
			break; //ncrunch: no coverage end
		}
	}

	private static (Dictionary<int, string> Labels, Dictionary<int, int> JumpEndPositions)
		BuildJumpLabels(List<Instruction> instructions)
	{
		var labels = new Dictionary<int, string>();
		var jumpEndPositions = new Dictionary<int, int>();
		var labelIndex = 0;
		for (var index = 0; index < instructions.Count; index++)
		{
			switch (instructions[index])
			{
			case JumpToId { InstructionType: InstructionType.JumpEnd } jumpEnd:
				jumpEndPositions[jumpEnd.Id] = index;
				AddLabelAt(labels, index, ref labelIndex);
				break;
			case Jump jump:
				AddLabelAt(labels, index + jump.InstructionsToSkip + 1, ref labelIndex);
				break;
			}
		}
		return (labels, jumpEndPositions);
	}

	private static void AddLabelAt(Dictionary<int, string> labels, int target, ref int labelIndex)
	{
		if (target >= 0 && !labels.ContainsKey(target))
			labels[target] = $"L{labelIndex++}";
	}

	private static void EmitInstruction(Instruction instruction, List<string> lines,
		Dictionary<string, int> paramIndexByName, Dictionary<string, int> variableSlots,
		List<(string Label, double Value)> dataConstants,
		List<(string Label, string Text)> printStrings,
		Dictionary<int, string> jumpLabels,
		Dictionary<int, int> jumpEndPositions, List<Instruction> allInstructions, int index,
		Platform platform = Platform.Linux,
		Dictionary<Register, List<Expression>> registerInstances = null!,
		Dictionary<string, List<Expression>> variableInstances = null!,
		Dictionary<string, CompiledMethodInfo>? compiledMethods = null,
		HashSet<int>? optimizedReturns = null)
	{
		switch (instruction)
		{
		case StoreVariableInstruction storeConst
			when !paramIndexByName.ContainsKey(storeConst.Identifier):
			//ncrunch: no coverage start
			if (variableSlots.TryGetValue(storeConst.Identifier, out var storeSlot))
				EmitStoreConstantToSlot(storeConst.ValueInstance, storeSlot, dataConstants, lines);
			break; //ncrunch: no coverage end
		case StoreVariableInstruction:
			break;
		case LoadVariableToRegister loadVar:
			if (variableInstances.TryGetValue(loadVar.Identifier, out var loadedInstance))
			{ //ncrunch: no coverage start
				registerInstances[loadVar.Register] = loadedInstance;
				break;
			} //ncrunch: no coverage end
			if (paramIndexByName.TryGetValue(loadVar.Identifier, out var paramIndex))
			{
				var sourceXmm = "xmm" + paramIndex;
				var destinationXmm = ToXmm(loadVar.Register);
				if (sourceXmm != destinationXmm)
					lines.Add("    movsd " + destinationXmm + ", " + sourceXmm);
				break;
			}
			if (variableSlots.TryGetValue(loadVar.Identifier, out var loadSlot))
				lines.Add("    movsd " + ToXmm(loadVar.Register) + ", [rbp-" + (loadSlot + 1) * 8 + "]");
			break;
		case LoadConstantInstruction loadConst:
			EmitLoadConstant(loadConst.Register, loadConst.Constant, dataConstants, lines);
			break;
		case BinaryInstruction binary when !binary.IsConditional():
			EmitArithmetic(binary, allInstructions, index, optimizedReturns, lines);
			break;
		case BinaryInstruction binary:
			EmitComparison(binary, lines);
			break;
		case StoreFromRegisterInstruction storeReg:
			if (registerInstances.TryGetValue(storeReg.Register, out var constructorArguments))
			{
				variableInstances[storeReg.Identifier] = constructorArguments;
				break;
			}
			if (variableSlots.TryGetValue(storeReg.Identifier, out var destinationSlot))
				lines.Add("    movsd [rbp-" + (destinationSlot + 1) * 8 + "], " + ToXmm(storeReg.Register));
			break;
		case ReturnInstruction ret:
			if (optimizedReturns != null && optimizedReturns.Contains(index))
				break;
			var src = ToXmm(ret.Register);
			if (src != "xmm0")
				lines.Add("    movsd xmm0, " + src);
			break;
		case PrintInstruction print:
			EmitPrint(print, printStrings, lines, platform);
			break;
		case Jump jump:
			EmitJump(jump, jumpLabels, index, lines);
			break;
		case Invoke invoke:
			EmitInvoke(invoke, lines, paramIndexByName, variableSlots, dataConstants,
				registerInstances, variableInstances, compiledMethods);
			break;
		case JumpToId { InstructionType: InstructionType.JumpEnd }:
			break;
		case JumpToId jumpToId:
			EmitJumpToId(jumpToId, jumpEndPositions, jumpLabels, allInstructions, index, lines);
			break;
		}
	}

	private static void EmitInvoke(Invoke invoke, List<string> lines,
		Dictionary<string, int> paramIndexByName, Dictionary<string, int> variableSlots,
		List<(string Label, double Value)> dataConstants,
		Dictionary<Register, List<Expression>> registerInstances,
		Dictionary<string, List<Expression>> variableInstances,
		Dictionary<string, CompiledMethodInfo>? compiledMethods)
	{
		if (invoke.Method == null)
			throw new NotSupportedException("Invoke instruction is missing method metadata"); //ncrunch: no coverage
		if (invoke.Method.Method.Name == Method.From && invoke.Method.Instance == null)
		{
			registerInstances[invoke.Register] = ResolveConstructorArguments(invoke.Method);
			return;
		}
		var methodKey = BuildMethodHeaderKeyInternal(invoke.Method.Method);
		if (compiledMethods == null || !compiledMethods.TryGetValue(methodKey, out var methodInfo))
			throw new NotSupportedException( //ncrunch: no coverage
				"Non-print method calls cannot be compiled to native assembly. " +
				"Use the interpreted runner for programs with complex runtime method calls.");
		var argumentExpressions = new List<Expression>();
		if (methodInfo.MemberNames.Count > 0)
			argumentExpressions.AddRange(ResolveInstanceMemberArguments(invoke.Method, variableInstances));
		for (var argumentIndex = 0; argumentIndex < invoke.Method.Arguments.Count; argumentIndex++)
			argumentExpressions.Add(invoke.Method.Arguments[argumentIndex]);
		if (argumentExpressions.Count > 8)
			throw new NotSupportedException( //ncrunch: no coverage
				"Native assembly compiler currently supports up to 8 call arguments");
		for (var argumentIndex = 0; argumentIndex < argumentExpressions.Count; argumentIndex++)
			EmitLoadExpressionToXmm(argumentExpressions[argumentIndex], "xmm" + argumentIndex, lines,
				paramIndexByName, variableSlots, dataConstants, variableInstances);
		lines.Add("    call " + methodInfo.Symbol);
		var destination = ToXmm(invoke.Register);
		if (destination != "xmm0")
			lines.Add("    movsd " + destination + ", xmm0");
	}

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

	private static void EmitLoadExpressionToXmm(Expression expression, string destinationXmm,
		List<string> lines, Dictionary<string, int> paramIndexByName,
		Dictionary<string, int> variableSlots,
		List<(string Label, double Value)> dataConstants,
		Dictionary<string, List<Expression>> variableInstances)
	{
		if (expression is Value value)
		{
			if (value.Data.IsText)
				throw new NotSupportedException( //ncrunch: no coverage
					"Text call arguments are not supported in native assembly calls");
			if (value.Data.Number == 0.0)
				lines.Add("    xorpd " + destinationXmm + ", " + destinationXmm); //ncrunch: no coverage
			else
				lines.Add("    movsd " + destinationXmm + ", [rel " +
					GetOrAddConstantLabel(value.Data.Number, dataConstants) + "]");
			return;
		}
		//ncrunch: no coverage start
		var variableName = expression.ToString();
		if (paramIndexByName.TryGetValue(variableName, out var parameterIndex))
		{
			var sourceXmm = "xmm" + parameterIndex;
			if (sourceXmm != destinationXmm)
				lines.Add("    movsd " + destinationXmm + ", " + sourceXmm);
			return;
		}
		if (variableSlots.TryGetValue(variableName, out var slot))
		{
			lines.Add("    movsd " + destinationXmm + ", [rbp-" + (slot + 1) * 8 + "]");
			return;
		}
		if (variableInstances.ContainsKey(variableName))
			throw new NotSupportedException("Cannot pass instance variable as numeric argument: " + variableName);
		throw new NotSupportedException("Unsupported call argument expression for native compilation: " + expression);
	} //ncrunch: no coverage end

	private static string GetOrAddConstantLabel(double number,
		List<(string Label, double Value)> dataConstants)
	{
		for (var index = 0; index < dataConstants.Count; index++)
			if (dataConstants[index].Value == number)
				return dataConstants[index].Label;
		//ncrunch: no coverage start
		var label = "const_" + dataConstants.Count;
		dataConstants.Add((label, number));
		return label;
	} //ncrunch: no coverage end

	private static void EmitPrint(PrintInstruction print,
		List<(string Label, string Text)> printStrings,
		List<string> lines, Platform platform)
	{
		var (strLabel, _) = printStrings.First(p => p.Text == BuildPrintKey(print));
		if (print.ValueRegister.HasValue && !print.ValueIsText)
		{
			var numXmm = ToXmm(print.ValueRegister.Value);
			if (platform == Platform.Windows)
			{
				if (print.TextPrefix.Length > 0)
				{
					lines.Add("    sub rsp, 16");
					lines.Add("    movsd [rsp], " + numXmm);
					EmitWindowsWriteFromLabel(strLabel, print.TextPrefix.Length, lines);
					lines.Add("    movsd xmm0, [rsp]");
					lines.Add("    add rsp, 16");
					EmitWindowsWriteNumberFromXmm("xmm0", lines);
				}
				else
					EmitWindowsWriteNumberFromXmm(numXmm, lines); //ncrunch: no coverage
				return;
			}
			//ncrunch: no coverage start
			lines.Add($"    lea rdi, [rel {strLabel}]");
			if (numXmm != "xmm0")
				lines.Add($"    movsd xmm0, {numXmm}");
			lines.Add("    mov eax, 1");
			lines.Add("    call printf");
			return;
		}
		if (platform == Platform.Windows)
		{
			EmitWindowsWriteFromLabel(strLabel, BuildPrintKey(print).Length + 1, lines);
			return;
		}
		lines.Add($"    lea rdi, [rel {strLabel}]");
		lines.Add("    xor eax, eax");
		lines.Add("    call printf");
	} //ncrunch: no coverage end

	private static void EmitWindowsWriteFromLabel(string label, int length, List<string> lines)
	{
		if (length <= 0)
			return; //ncrunch: no coverage
		lines.Add("    sub rsp, 48");
		lines.Add("    mov ecx, -11");
		lines.Add("    call GetStdHandle");
		lines.Add("    mov rcx, rax");
		lines.Add("    lea rdx, [rel " + label + "]");
		lines.Add("    mov r8d, " + length);
		lines.Add("    lea r9, [rsp+40]");
		lines.Add("    mov qword [rsp+32], 0");
		lines.Add("    call WriteFile");
		lines.Add("    add rsp, 48");
	}

	private static void EmitWindowsWriteNumberFromXmm(string sourceXmm, List<string> lines)
	{
		if (sourceXmm != "xmm0")
			lines.Add("    movsd xmm0, " + sourceXmm); //ncrunch: no coverage
		lines.Add("    call print_number_from_xmm");
	}

	private static string BuildWindowsPrintNumberHelper() =>
		string.Join("\n", "", "section .text", "print_number_from_xmm:", "    push rbp",
			"    mov rbp, rsp", "    sub rsp, 96", "    movsd [rsp], xmm0", "    mov ecx, -11",
			"    call GetStdHandle", "    mov rcx, rax", "    lea r10, [rsp+79]",
			"    mov byte [r10], 10", "    mov r11, r10", "    movsd xmm0, [rsp]",
			"    cvttsd2si rax, xmm0", "    xor r9d, r9d", "    test rax, rax",
			"    jge .print_abs_done", "    mov r9d, 1", "    neg rax", ".print_abs_done:",
			"    test rax, rax", "    jne .print_digits_loop", "    dec r11",
			"    mov byte [r11], '0'", "    jmp .print_digits_done", ".print_digits_loop:",
			"    xor edx, edx", "    mov r8, 10", "    div r8", "    add dl, '0'", "    dec r11",
			"    mov [r11], dl", "    test rax, rax", "    jne .print_digits_loop",
			".print_digits_done:", "    test r9d, r9d", "    je .print_sign_done",
			"    dec r11", "    mov byte [r11], '-'", ".print_sign_done:", "    mov rdx, r11",
			"    mov r8, r10", "    sub r8, r11", "    inc r8", "    lea r9, [rsp+40]",
			"    mov qword [rsp+32], 0", "    call WriteFile", "    add rsp, 96", "    pop rbp",
			"    ret");

	private static string BuildPrintKey(PrintInstruction print) =>
		print.ValueRegister.HasValue && !print.ValueIsText
			? print.TextPrefix + "%g"
			: print.TextPrefix;

	private static List<(string Label, string Text)> CollectPrintStrings(List<Instruction> instructions)
	{
		var strings = new List<(string, string)>();
		var seen = new HashSet<string>(StringComparer.Ordinal);
		var labelIndex = 0;
		foreach (var instruction in instructions.OfType<PrintInstruction>())
		{
			var key = BuildPrintKey(instruction);
			if (seen.Add(key))
				strings.Add(($"str_{labelIndex++}", key));
		}
		return strings;
	}

	private static string BuildStringBytes(string text)
	{
		if (text.Length == 0)
			return ""; //ncrunch: no coverage
		var parts = new List<string>();
		var ascii = new System.Text.StringBuilder();
		foreach (var c in text)
		{
			if (c is >= ' ' and <= '~' && c != '"' && c != '\\')
				ascii.Append(c);
			else
			{ //ncrunch: no coverage start
				if (ascii.Length > 0)
				{
					parts.Add($"\"{ascii}\"");
					ascii.Clear();
				}
				parts.Add(((int)c).ToString());
			} //ncrunch: no coverage end
		}
		if (ascii.Length > 0)
			parts.Add($"\"{ascii}\"");
		return string.Join(", ", parts);
	}

	//ncrunch: no coverage start
	private static void EmitStoreConstantToSlot(ValueInstance value, int slot,
		List<(string Label, double Value)> dataConstants, List<string> lines)
	{
		if (value.IsText)
			return;
		var number = value.Number;
		if (number == 0.0)
		{
			lines.Add("    xorpd xmm15, xmm15");
			lines.Add($"    movsd [rbp-{(slot + 1) * 8}], xmm15");
		}
		else
		{
			var constLabel = dataConstants.First(c => c.Value == number).Label;
			lines.Add($"    movsd xmm15, [rel {constLabel}]");
			lines.Add($"    movsd [rbp-{(slot + 1) * 8}], xmm15");
		}
	} //ncrunch: no coverage end

	private static void EmitLoadConstant(Register register, ValueInstance value,
		List<(string Label, double Value)> dataConstants, List<string> lines)
	{
		var dest = ToXmm(register);
		if (value.IsText)
			return;
		if (value.Number == 0.0)
			lines.Add($"    xorpd {dest}, {dest}");
		else
		{
			var constLabel = dataConstants.First(c => c.Value == value.Number).Label;
			lines.Add($"    movsd {dest}, [rel {constLabel}]");
		}
	}

	private static void EmitArithmetic(BinaryInstruction binary, List<Instruction> allInstructions,
		int instructionIndex, HashSet<int>? optimizedReturns, List<string> lines)
	{
		var src0 = ToXmm(binary.Registers[0]);
		var src1 = ToXmm(binary.Registers[1]);
		var dest = ToXmm(binary.Registers[^1]);
		var op = binary.InstructionType switch
		{
			InstructionType.Add => "addsd",
			InstructionType.Subtract => "subsd",
			InstructionType.Multiply => "mulsd",
			InstructionType.Divide => "divsd",
			InstructionType.Modulo => null, //ncrunch: no coverage
			_ => throw new NotSupportedException( //ncrunch: no coverage
				$"x64 compilation of {binary.InstructionType} is not supported")
		};
		if (op == null)
			return; //ncrunch: no coverage
		if (instructionIndex + 1 < allInstructions.Count &&
			allInstructions[instructionIndex + 1] is ReturnInstruction returnInstruction &&
			returnInstruction.Register == binary.Registers[^1] && src0 == "xmm0")
		{
			lines.Add("    " + op + " xmm0, " + src1);
			optimizedReturns?.Add(instructionIndex + 1);
			return;
		}
		if (dest != src0)
			lines.Add("    movsd " + dest + ", " + src0);
		lines.Add("    " + op + " " + dest + ", " + src1);
	}

	private static void EmitComparison(BinaryInstruction binary, List<string> lines)
	{
		var src0 = ToXmm(binary.Registers[0]);
		var src1 = ToXmm(binary.Registers[1]);
		lines.Add($"    ucomisd {src0}, {src1}");
	}

	private static void EmitJump(Jump jump, Dictionary<int, string> jumpLabels, int index,
		List<string> lines)
	{
		var target = index + jump.InstructionsToSkip + 1;
		var label = jumpLabels.TryGetValue(target, out var lbl)
			? $".{lbl}"
			: $".unknown_{target}";
		var op = jump.InstructionType switch
		{
			InstructionType.JumpIfTrue => "je",
			InstructionType.JumpIfFalse => "jne",
			_ => "jmp"
		};
		lines.Add($"    {op} {label}");
	}

	private static void EmitJumpToId(JumpToId jumpToId, Dictionary<int, int> jumpEndPositions,
		Dictionary<int, string> jumpLabels, List<Instruction> allInstructions, int index,
		List<string> lines)
	{
		if (!jumpEndPositions.TryGetValue(jumpToId.Id, out var endIndex) ||
			!jumpLabels.TryGetValue(endIndex, out var label))
			return; //ncrunch: no coverage
		var prevComparison = index > 0
			? allInstructions[index - 1] as BinaryInstruction
			: null;
		var op = jumpToId.InstructionType switch
		{
			InstructionType.JumpToIdIfFalse => GetFalseJumpOp(prevComparison?.InstructionType),
			InstructionType.JumpToIdIfTrue => GetTrueJumpOp(prevComparison?.InstructionType), //ncrunch: no coverage
			_ => "jmp" //ncrunch: no coverage
		};
		lines.Add($"    {op} .{label}");
	}

	private static string GetFalseJumpOp(InstructionType? comparisonType) =>
		comparisonType switch
		{
			InstructionType.Equal => "jne",
			InstructionType.NotEqual => "je", //ncrunch: no coverage
			InstructionType.LessThan => "jae", //ncrunch: no coverage
			InstructionType.GreaterThan => "jbe",
			_ => "jne" //ncrunch: no coverage
		};

	//ncrunch: no coverage start
	private static string GetTrueJumpOp(InstructionType? comparisonType) =>
		comparisonType switch
		{
			InstructionType.Equal => "je",
			InstructionType.NotEqual => "jne",
			InstructionType.LessThan => "jb",
			InstructionType.GreaterThan => "ja",
			_ => "je"
		}; //ncrunch: no coverage end

	private static string ToXmm(Register register) => $"xmm{(int)register}";
}