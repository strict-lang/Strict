using System.Globalization;
using System.Text;
using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Language;

namespace Strict.Compiler.Assembly;

public sealed partial class InstructionsToLlvmIr
{
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

	private static void EmitReturn(ReturnInstruction ret, List<string> lines, EmitContext context)
	{
		var value = GetRegisterValue(ret.Register, context);
		lines.Add($"  ret double {value}");
		context.HasReturn = true;
		context.TerminatedBlocks.Add(context.CurrentBlock);
	}

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
			throw new NotSupportedByBackend(
				"Invoke instruction is missing method metadata"); //ncrunch: no coverage
		if (invoke.MethodInfo.MethodName == Method.From && !invoke.MethodInfo.InstanceRegister.HasValue)
		{
			context.RegisterInstances[invoke.Register] = invoke.MethodInfo.ArgumentRegisters;
			return;
		}
		var methodKey = BuildMethodHeaderKeyInternal(invoke.MethodInfo);
		if (context.CompiledMethods == null ||
			!context.CompiledMethods.TryGetValue(methodKey, out var methodInfo))
			throw new NotSupportedByBackend( //ncrunch: no coverage
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
}
