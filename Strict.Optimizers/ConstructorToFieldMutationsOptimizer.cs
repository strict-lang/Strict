using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Language;

namespace Strict.Optimizers;

/// <summary>
/// Replaces <c>Invoke(Color.from(expr1, expr2, expr3))</c> patterns (produced by
/// <see cref="MethodInliningOptimizer"/> inlining GetBrightnessAdjustedColor) with
/// <see cref="ConstructValueTypeInstruction"/> so the VirtualMachine avoids a full
/// method-dispatch constructor call in the hot loop.
/// The replacement reuses the registers that already hold the computed field values
/// in the instruction stream just before the Invoke — no new register allocation needed.
/// </summary>
public sealed class ConstructorToFieldMutationsOptimizer : InstructionOptimizer
{
	public override void Optimize(BinaryExecutable binary)
	{
		foreach (var typeEntry in binary.MethodsPerType)
		foreach (var methodGroup in typeEntry.Value.MethodGroups.Values)
		foreach (var method in methodGroup)
			method.instructions = Optimize(method.instructions, binary);
	}

	public override List<Instruction> Optimize(List<Instruction> instructions) => instructions;

	private static List<Instruction> Optimize(List<Instruction> instructions,
		BinaryExecutable binary)
	{
		for (var index = 0; index < instructions.Count; index++)
		{
			if (instructions[index] is Invoke invoke &&
				TryBuildConstructValueType(invoke, instructions, index, binary, out var replacement))
				instructions[index] = replacement;
		}
		return instructions;
	}

	private static bool TryBuildConstructValueType(Invoke invoke, List<Instruction> instructions,
		int invokeIndex, BinaryExecutable binary, out ConstructValueTypeInstruction replacement)
	{
		replacement = null!;
		if (invoke.MethodInfo.MethodName != Method.From || invoke.MethodInfo.InstanceRegister.HasValue)
			return false;
		var argRegisters = invoke.MethodInfo.ArgumentRegisters;
		if (argRegisters.Length == 0)
			return false;
		var returnType = invoke.MethodInfo.ResolveReturnType(binary.basePackage);
		if (returnType.IsNumber || returnType.IsText || returnType.IsBoolean || returnType.IsList ||
			returnType.IsDictionary || returnType.IsTrait || returnType.Members.Count == 0)
			return false;
		if (argRegisters.Length > returnType.Members.Count)
			return false;
		if (!AllPrecedingArgsAreBinaryInstructions(instructions, invokeIndex, argRegisters.Length))
			return false;
		replacement = new ConstructValueTypeInstruction(invoke.Register, returnType, argRegisters);
		return true;
	}

	private static bool AllPrecedingArgsAreBinaryInstructions(List<Instruction> instructions,
		int invokeIndex, int argCount)
	{
		var remaining = argCount;
		for (var scan = invokeIndex - 1; scan >= 0 && remaining > 0; scan--)
		{
			if (instructions[scan] is not BinaryInstruction bin || bin.Registers.Length < 3 ||
				bin.IsConditional())
				continue;
			remaining--;
		}
		return remaining == 0;
	}
}