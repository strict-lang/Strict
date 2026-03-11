using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Compiler.X64;

/// <summary>
/// Compiles a Strict method or pre-compiled instruction list to x64 NASM assembly text.
/// Strict registers R0–R15 map to XMM registers xmm0–xmm15 for numeric (double) values.
/// Follows the System V AMD64 ABI: first 8 float/double parameters in xmm0–xmm7, return in xmm0.
/// The generated NASM text can be assembled with: nasm -f win64 output.asm -o output.obj
/// </summary>
public sealed class InstructionsToX64Compiler
{
	public string Compile(Method method)
	{
		var paramNames = method.Parameters.Select(p => p.Name);
		return BuildX64Assembly(method.Name, paramNames, GenerateInstructions(method));
	}

	public string CompileInstructions(string methodName, IList<Instruction> instructions) =>
		BuildX64Assembly(methodName, [], [.. instructions]);

	private static List<Instruction> GenerateInstructions(Method method)
	{
		var body = method.GetBodyAndParseIfNeeded();
		var expressions = body is Body b ? b.Expressions : [body];
		var arguments = method.Parameters.ToDictionary(p => p.Name, p => new ValueInstance(p.Type, 0));
		return new BytecodeGenerator(new InvokedMethod(expressions, arguments, method.ReturnType),
			new Registry()).Generate();
	}

	private static string BuildX64Assembly(string methodName, IEnumerable<string> paramNames,
		List<Instruction> instructions)
	{
		var paramIndexByName = paramNames
			.Select((name, index) => (name, index))
			.ToDictionary(x => x.name, x => x.index);
		var variableSlots = BuildVariableSlots(paramIndexByName.Keys, instructions);
		var dataConstants = CollectConstants(instructions);
		var (jumpLabels, jumpEndPositions) = BuildJumpLabels(instructions);
		var lines = new List<string>();
		if (dataConstants.Count > 0)
		{
			lines.Add("section .data");
			foreach (var (label, value) in dataConstants)
				lines.Add($"    {label}: dq 0x{BitConverter.DoubleToInt64Bits(value):X16}");
			lines.Add("");
		}
		lines.Add("section .text");
		lines.Add($"global {methodName}");
		lines.Add("");
		lines.Add($"{methodName}:");
		lines.Add("    push rbp");
		lines.Add("    mov rbp, rsp");
		var frameSize = AlignTo16(variableSlots.Count * 8);
		if (frameSize > 0)
			lines.Add($"    sub rsp, {frameSize}");
		foreach (var (name, slot) in variableSlots.Where(kv => paramIndexByName.ContainsKey(kv.Key)))
		{
			var paramIndex = paramIndexByName[name];
			lines.Add($"    movsd [rbp-{(slot + 1) * 8}], xmm{paramIndex}");
		}
		for (var index = 0; index < instructions.Count; index++)
		{
			if (jumpLabels.TryGetValue(index, out var label))
				lines.Add($".{label}:");
			EmitInstruction(instructions[index], lines, paramIndexByName, variableSlots,
				dataConstants, jumpLabels, jumpEndPositions, instructions, index);
		}
		if (frameSize > 0)
			lines.Add($"    add rsp, {frameSize}");
		lines.Add("    pop rbp");
		lines.Add("    ret");
		return string.Join("\n", lines);
	}

	private static int AlignTo16(int size) => (size + 15) / 16 * 16;

	private static Dictionary<string, int> BuildVariableSlots(IEnumerable<string> paramNames,
		List<Instruction> instructions)
	{
		var slots = new Dictionary<string, int>();
		foreach (var name in paramNames)
			slots[name] = slots.Count;
		foreach (var instruction in instructions)
		{
			var varName = instruction switch
			{
				StoreFromRegisterInstruction store => store.Identifier,
				StoreVariableInstruction store => store.Identifier,
				_ => null
			};
			if (varName != null && !slots.ContainsKey(varName))
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
				LoadConstantInstruction load when !load.ValueInstance.IsText =>
					(double?)load.ValueInstance.Number,
				StoreVariableInstruction store when !store.ValueInstance.IsText =>
					store.ValueInstance.Number,
				_ => null
			};
			if (value is { } v && v != 0.0 && seenValues.Add(v))
				constants.Add(($"const_{index++}", v));
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
		{
			switch (instructions[index])
			{
			case Jump jump:
				AddLabelAt(labels, index + jump.InstructionsToSkip + 1, ref labelIndex);
				break;
			case JumpIf jumpIf:
				AddLabelAt(labels, index + jumpIf.Steps + 1, ref labelIndex);
				break;
			case JumpToId { InstructionType: InstructionType.JumpEnd } jumpEnd:
				jumpEndPositions[jumpEnd.Id] = index;
				AddLabelAt(labels, index, ref labelIndex);
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
		List<(string Label, double Value)> dataConstants, Dictionary<int, string> jumpLabels,
		Dictionary<int, int> jumpEndPositions, List<Instruction> allInstructions, int index)
	{
		switch (instruction)
		{
		case StoreVariableInstruction storeConst when !paramIndexByName.ContainsKey(storeConst.Identifier):
			EmitStoreConstantToSlot(storeConst.ValueInstance, variableSlots[storeConst.Identifier],
				dataConstants, lines);
			break;
		case StoreVariableInstruction:
			break;
		case LoadVariableToRegister loadVar:
			lines.Add($"    movsd {ToXmm(loadVar.Register)}, [rbp-{(variableSlots[loadVar.Identifier] + 1) * 8}]");
			break;
		case LoadConstantInstruction loadConst:
			EmitLoadConstant(loadConst.Register, loadConst.ValueInstance, dataConstants, lines);
			break;
		case BinaryInstruction binary when !binary.IsConditional():
			EmitArithmetic(binary, lines);
			break;
		case BinaryInstruction binary:
			EmitComparison(binary, lines);
			break;
		case StoreFromRegisterInstruction storeReg:
			lines.Add($"    movsd [rbp-{(variableSlots[storeReg.Identifier] + 1) * 8}], {ToXmm(storeReg.Register)}");
			break;
		case ReturnInstruction ret:
		{
			var src = ToXmm(ret.Register);
			if (src != "xmm0")
				lines.Add($"    movsd xmm0, {src}");
			break;
		}
		case Jump jump:
			EmitJump(jump, jumpLabels, index, lines);
			break;
		case JumpToId { InstructionType: InstructionType.JumpEnd }:
			break;
		case JumpToId jumpToId:
			EmitJumpToId(jumpToId, jumpEndPositions, jumpLabels, allInstructions, index, lines);
			break;
		}
	}

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
	}

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

	private static void EmitArithmetic(BinaryInstruction binary, List<string> lines)
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
			// x64 has no single SSE2 instruction for double modulo; fmod requires a runtime call
			InstructionType.Modulo => null,
			_ => throw new NotSupportedException($"x64 compilation of {binary.InstructionType} is not supported") //ncrunch: no coverage
		};
		if (op == null)
			return;
		if (dest != src0)
			lines.Add($"    movsd {dest}, {src0}");
		lines.Add($"    {op} {dest}, {src1}");
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
		var label = jumpLabels.TryGetValue(target, out var lbl) ? $".{lbl}" : $".unknown_{target}";
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
			return;
		var prevComparison = index > 0 ? allInstructions[index - 1] as BinaryInstruction : null;
		var op = jumpToId.InstructionType switch
		{
			InstructionType.JumpToIdIfFalse => GetFalseJumpOp(prevComparison?.InstructionType),
			InstructionType.JumpToIdIfTrue => GetTrueJumpOp(prevComparison?.InstructionType),
			_ => "jmp" //ncrunch: no coverage
		};
		lines.Add($"    {op} .{label}");
	}

	private static string GetFalseJumpOp(InstructionType? comparisonType) =>
		comparisonType switch
		{
			InstructionType.Equal => "jne",
			InstructionType.NotEqual => "je",
			InstructionType.LessThan => "jae",
			InstructionType.GreaterThan => "jbe",
			_ => "jne"
		};

	private static string GetTrueJumpOp(InstructionType? comparisonType) =>
		comparisonType switch
		{
			InstructionType.Equal => "je",
			InstructionType.NotEqual => "jne",
			InstructionType.LessThan => "jb",
			InstructionType.GreaterThan => "ja",
			_ => "je"
		};

	private static string ToXmm(Register register) => $"xmm{(int)register}";
}
