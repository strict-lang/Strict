using System.Text;
using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Assembly;

public sealed partial class InstructionsToAssembly
{
	private static void AddLabelAt(Dictionary<int, string> labels, int target, ref int labelIndex)
	{
		if (target >= 0 && !labels.ContainsKey(target))
			labels[target] = $"L{labelIndex++}";
	}

	private static void EmitInvoke(Invoke invoke, List<string> lines,
		Dictionary<Register, Register[]> registerInstances,
		Dictionary<string, CompiledMethodInfo>? compiledMethods)
	{
		if (invoke.MethodInfo == null)
			throw new NotSupportedByBackend(
				"Invoke instruction is missing method metadata"); //ncrunch: no coverage
		if (IsFileRuntimeInvoke(invoke.MethodInfo))
		{
			EmitFileRuntimeInvoke(invoke.MethodInfo, lines);
			return;
		}
		if (invoke.MethodInfo.MethodName == Method.From && !invoke.MethodInfo.InstanceRegister.HasValue)
		{
			registerInstances[invoke.Register] = invoke.MethodInfo.ArgumentRegisters;
			return;
		}
		var methodKey = BuildMethodHeaderKeyInternal(invoke.MethodInfo);
		if (compiledMethods == null || !compiledMethods.TryGetValue(methodKey, out var methodInfo))
			throw new NotSupportedByBackend( //ncrunch: no coverage
				"Non-print method calls cannot be compiled to native assembly. " +
				"Use the interpreted runner for programs with complex runtime method calls.");
		var sourceRegisters = new List<Register>();
		if (methodInfo.MemberNames.Count > 0 && invoke.MethodInfo.InstanceRegister.HasValue &&
			registerInstances.TryGetValue(invoke.MethodInfo.InstanceRegister.Value,
				out var memberRegisters))
			sourceRegisters.AddRange(memberRegisters);
		sourceRegisters.AddRange(invoke.MethodInfo.ArgumentRegisters);
		if (sourceRegisters.Count > 8)
			throw new NotSupportedByBackend( //ncrunch: no coverage
				"Native assembly compiler currently supports up to 8 call arguments");
		for (var argumentIndex = 0; argumentIndex < sourceRegisters.Count; argumentIndex++)
		{
			var sourceXmm = ToXmm(sourceRegisters[argumentIndex]);
			var destinationXmm = "xmm" + argumentIndex;
			if (sourceXmm != destinationXmm)
				lines.Add("    movsd " + destinationXmm + ", " + sourceXmm);
		}
		lines.Add("    call " + methodInfo.Symbol);
		var destination = ToXmm(invoke.Register);
		if (destination != "xmm0")
			lines.Add("    movsd " + destination + ", xmm0");
	}

	private static bool IsFileRuntimeInvoke(InvokeMethodInfo info) =>
		(info.TypeFullName == Type.File ||
			info.TypeFullName.EndsWith(Context.ParentSeparator + Type.File, StringComparison.Ordinal)) &&
		info.MethodName is Method.From or "Write" or "ReadLines" or "ReadBytes" or "Close" or "Length"
			or "Exists";

	private static void EmitFileRuntimeInvoke(InvokeMethodInfo info, List<string> lines) =>
		lines.Add("    call strict_file_" + info.MethodName switch
		{
			Method.From => "open",
			"Write" when info.ParameterNames.Length > 0 && info.ParameterNames[0].
				Contains("bytes", StringComparison.OrdinalIgnoreCase) => "write_bytes",
			"Write" => "write_text",
			"ReadLines" => "read_lines",
			"ReadBytes" => "read_bytes",
			"Close" => "close",
			"Length" => "length",
			_ => "exists"
		});

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
			InstructionType.JumpToIdIfTrue =>
				GetTrueJumpOp(prevComparison?.InstructionType), //ncrunch: no coverage
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
}
