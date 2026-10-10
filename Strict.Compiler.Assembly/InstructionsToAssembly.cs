using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;

namespace Strict.Compiler.Assembly;

/// <summary>
/// Compiles a Strict method or pre-compiled instruction list to 64 bit NASM assembly text.
/// Strict registers R0–R15 map to XMM registers xmm0–xmm15 for numeric (double) values.
/// Follows the System V AMD64 ABI: first 8 float/double parameters in xmm0–xmm7, return in xmm0.
/// The generated NASM text can be assembled with: nasm -f win64 output.asm -o output.obj
/// </summary>
public sealed partial class InstructionsToAssembly : InstructionsCompiler
{
	public override Task<string> Compile(BinaryExecutable binary, Platform platform)
	{
		var precompiledMethods = BuildPrecompiledMethodsInternal(binary);
		var output = CompileForPlatform(Method.Run, binary.EntryPoint.instructions, platform,
			precompiledMethods, binary);
		return Task.FromResult(output);
	}

	public override string Extension => ".asm";

	public string CompileInstructions(string methodName, List<Instruction> instructions) =>
		BuildAssembly(methodName, [], instructions);

	private static string CompileForPlatform(string methodName,
		IReadOnlyList<Instruction> instructions, Platform platform,
		IReadOnlyDictionary<string, List<Instruction>>? precompiledMethods = null,
		BinaryExecutable? binary = null)
	{
		var hasPrint = instructions.OfType<PrintInstruction>().Any();
		var methodInfos = CollectMethods([.. instructions], precompiledMethods, binary);
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

	private static string
		BuildEntryPoint(string methodName, Platform platform, bool hasPrint = false) =>
		platform switch
		{
			Platform.Windows => BuildWindowsEntryPoint(methodName, hasPrint),
			Platform.Linux => BuildLinuxEntryPoint(methodName, hasPrint),
			Platform.MacOS => BuildMacOsEntryPoint(methodName, hasPrint),
			_ => throw new UnsupportedPlatform(platform) //ncrunch: no coverage
		};

	private static string BuildWindowsEntryPoint(string methodName, bool hasPrint) =>
		string.Join("\n", "", "extern ExitProcess", hasPrint
				? "extern GetStdHandle"
				: "", hasPrint
				? "extern WriteFile"
				: "", "", "global main", "", "main:", "    push rbp", "    mov rbp, rsp", "    sub rsp, 32",
			$"    call {methodName}", "    xor rcx, rcx", "    call ExitProcess", "    add rsp, 32",
			"    pop rbp", "    ret");

	private static string BuildLinuxEntryPoint(string methodName, bool hasPrint)
	{
		if (hasPrint)
			return string.Join("\n", "extern printf", "", "global main", "",
				"main:", //ncrunch: no coverage
				"    push rbp", "    mov rbp, rsp", $"    call {methodName}", "    mov rdi, 0",
				"    mov rax, 60", "    syscall");
		return string.Join("\n", "", "global _start", "", "_start:", "    push rbp", "    mov rbp, rsp",
			$"    call {methodName}", "    mov rdi, 0", "    mov rax, 60", "    syscall");
	}

	private static string BuildMacOsEntryPoint(string methodName, bool hasPrint)
	{
		var printExtern = hasPrint
			? "extern _printf\n"
			: "";
		return printExtern + string.Join("\n", "", "global _main", "", "_main:", "    push rbp",
			"    mov rbp, rsp", $"    call _{methodName}", "    xor rdi, rdi", "    mov rax, 0x2000001",
			"    syscall");
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
		var registerInstances = new Dictionary<Register, Register[]>();
		var variableInstances = new Dictionary<string, Register[]>(StringComparer.Ordinal);
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

	private static bool NeedsStackFrame(int frameSize, IEnumerable<Instruction> instructions) =>
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

	private static List<(string Label, double Value)> CollectConstants(List<Instruction> instructions)
	{
		var constants = new List<(string, double)>();
		var seenValues = new HashSet<double>();
		var index = 0;
		foreach (var instruction in instructions)
		{
			var value = instruction switch
			{
				LoadConstantInstruction load when !load.Constant.IsText => (double?)load.Constant.Number,
				StoreVariableInstruction store when !store.ValueInstance.IsText => store.ValueInstance.
					Number,
				_ => null
			};
			if (value is { } constantValue && constantValue != 0.0 && seenValues.Add(constantValue))
				constants.Add(($"const_{index++}", constantValue));
		}
		return constants;
	}

	private static (Dictionary<int, string> Labels, Dictionary<int, int> JumpEndPositions)
		BuildJumpLabels(List<Instruction> instructions)
	{
		var labels = new Dictionary<int, string>();
		var jumpEndPositions = new Dictionary<int, int>();
		var labelIndex = 0;
		for (var index = 0; index < instructions.Count; index++)
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
		return (labels, jumpEndPositions);
	}

	private static void EmitInstruction(Instruction instruction, List<string> lines,
		Dictionary<string, int> paramIndexByName, Dictionary<string, int> variableSlots,
		IEnumerable<(string Label, double Value)> dataConstants,
		IEnumerable<(string Label, string Text)> printStrings, Dictionary<int, string> jumpLabels,
		Dictionary<int, int> jumpEndPositions, List<Instruction> allInstructions, int index,
		Platform platform = Platform.Linux, Dictionary<Register, Register[]> registerInstances = null!,
		Dictionary<string, Register[]> variableInstances = null!,
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
			EmitInvoke(invoke, lines, registerInstances, compiledMethods);
			break;
		case JumpToId { InstructionType: InstructionType.JumpEnd }:
			break;
		case JumpToId jumpToId:
			EmitJumpToId(jumpToId, jumpEndPositions, jumpLabels, allInstructions, index, lines);
			break;
		}
	}

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

	//ncrunch: no coverage start
	private static void EmitStoreConstantToSlot(ValueInstance value, int slot,
		IEnumerable<(string Label, double Value)> dataConstants, List<string> lines)
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
		IEnumerable<(string Label, double Value)> dataConstants, List<string> lines)
	{
		var dest = ToXmm(register);
		if (value.IsText)
			return;
		if (value.Number == 0.0)
		{
			lines.Add($"    xorpd {dest}, {dest}");
		}
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
			InstructionType.Subtract or InstructionType.Modulo => "subsd",
			InstructionType.Multiply => "mulsd",
			InstructionType.Divide => "divsd",
			_ => throw new NotSupportedByBackend( //ncrunch: no coverage
				$"x64 compilation of {binary.InstructionType} is not supported")
		};
		if (binary.InstructionType == InstructionType.Modulo)
			src1 = EmitTruncatedQuotientTimesDivisor(src0, src1, lines);
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

	/// <summary>
	/// Strict % truncates like C#: a % b = a - trunc(a / b) * b, the product is left in xmm15.
	/// </summary>
	private static string EmitTruncatedQuotientTimesDivisor(string dividend, string divisor,
		List<string> lines)
	{
		lines.Add("    movsd xmm15, " + dividend);
		lines.Add("    divsd xmm15, " + divisor);
		lines.Add("    roundsd xmm15, xmm15, 3");
		lines.Add("    mulsd xmm15, " + divisor);
		return "xmm15";
	}

	private static void EmitComparison(BinaryInstruction binary, List<string> lines)
	{
		var src0 = ToXmm(binary.Registers[0]);
		var src1 = ToXmm(binary.Registers[1]);
		lines.Add($"    ucomisd {src0}, {src1}");
	}

	private static string ToXmm(Register register) => $"xmm{(int)register}";
}
