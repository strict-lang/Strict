using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Compiler.X64;

/// <summary>
/// Compiles a Strict method to x64 NASM assembly using bytecode instructions.
/// Strict registers R0–R15 map directly to XMM registers xmm0–xmm15 for numeric (double) values.
/// Follows the System V AMD64 ABI: first 8 float/double parameters in xmm0–xmm7, return in xmm0.
/// </summary>
public sealed class InstructionsToX64Compiler
{
	public string Compile(Method method) => BuildX64Assembly(method, GenerateInstructions(method));

	private static List<Instruction> GenerateInstructions(Method method)
	{
		var body = method.GetBodyAndParseIfNeeded();
		var expressions = body is Body b ? b.Expressions : [body];
		var arguments = method.Parameters.ToDictionary(p => p.Name, p => new ValueInstance(p.Type, 0));
		return new BytecodeGenerator(new InvokedMethod(expressions, arguments, method.ReturnType),
			new Registry()).Generate();
	}

	private static string BuildX64Assembly(Method method, List<Instruction> instructions)
	{
		var paramIndexByName = method.Parameters
			.Select((p, i) => (p.Name, Index: i))
			.ToDictionary(x => x.Name, x => x.Index);
		var variableSlots = BuildVariableSlots(method, instructions);
		var dataConstants = CollectConstants(instructions);
		var jumpLabels = BuildJumpLabels(instructions);
		var lines = new List<string>();
		if (dataConstants.Count > 0)
		{
			lines.Add("section .data");
			foreach (var (label, value) in dataConstants)
				lines.Add($"    {label}: dq 0x{BitConverter.DoubleToInt64Bits(value):X16}");
			lines.Add("");
		}
		lines.Add("section .text");
		lines.Add($"global {method.Name}");
		lines.Add("");
		lines.Add($"{method.Name}:");
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
				dataConstants, jumpLabels, index);
		}
		if (frameSize > 0)
			lines.Add($"    add rsp, {frameSize}");
		lines.Add("    pop rbp");
		lines.Add("    ret");
		return string.Join("\n", lines);
	}

	private static int AlignTo16(int size) => (size + 15) / 16 * 16;

	private static Dictionary<string, int> BuildVariableSlots(Method method,
		List<Instruction> instructions)
	{
		var slots = new Dictionary<string, int>();
		foreach (var param in method.Parameters)
			slots[param.Name] = slots.Count;
		foreach (var instruction in instructions)
		{
			var name = instruction switch
			{
				StoreFromRegisterInstruction store => store.Identifier,
				StoreVariableInstruction store => store.Identifier,
				_ => null
			};
			if (name != null && !slots.ContainsKey(name))
				slots[name] = slots.Count;
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

	private static Dictionary<int, string> BuildJumpLabels(List<Instruction> instructions)
	{
		var labels = new Dictionary<int, string>();
		var labelIndex = 0;
		for (var index = 0; index < instructions.Count; index++)
		{
			var target = instructions[index] switch
			{
				Jump jump => index + jump.InstructionsToSkip + 1,
				JumpIf jumpIf => index + jumpIf.Steps + 1,
				_ => -1
			};
			if (target >= 0 && !labels.ContainsKey(target))
				labels[target] = $"L{labelIndex++}";
		}
		return labels;
	}

	private static void EmitInstruction(Instruction instruction, List<string> lines,
		Dictionary<string, int> paramIndexByName, Dictionary<string, int> variableSlots,
		List<(string Label, double Value)> dataConstants, Dictionary<int, string> jumpLabels,
		int index)
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
		{
			var dest = ToXmm(loadVar.Register);
			lines.Add($"    movsd {dest}, [rbp-{(variableSlots[loadVar.Identifier] + 1) * 8}]");
			break;
		}
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
			return; // Modulo not directly translatable to a single SSE2 instruction
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

	private static string ToXmm(Register register) => $"xmm{(int)register}";
}
