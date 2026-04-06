using Strict.Bytecode.Instructions;
using Type = Strict.Language.Type;

namespace Strict.Optimizers;

/// <summary>
/// Moves loop-invariant LoadVariableToRegister instructions before the LoopBeginInstruction.
/// A load is loop-invariant when the variable it reads is never written inside the loop body
/// (no StoreFromRegisterInstruction or StoreVariableInstruction with the same name).
/// This avoids re-loading constants like 'brightness' on every loop iteration.
/// </summary>
public sealed class LoopInvariantCodeMotionOptimizer : InstructionOptimizer
{
	public override List<Instruction> Optimize(List<Instruction> instructions)
	{
		var result = new List<Instruction>(instructions);
		var changed = true;
		while (changed)
		{
			changed = false;
			for (var index = 0; index < result.Count; index++)
			{
				if (result[index] is not LoopBeginInstruction loopBegin)
					continue;
				var loopEnd = FindLoopEnd(result, index);
				if (loopEnd < 0)
					continue;
				var writtenInLoop = CollectWrittenVariables(result, index + 1, loopEnd);
				// Scan forward; only hoist simple invariant loads (variable or constant)
				for (var bodyIndex = index + 1; bodyIndex < loopEnd; bodyIndex++)
				{
					if (result[bodyIndex] is not LoadVariableToRegister load)
						continue;
					if (writtenInLoop.Contains(load.Identifier) ||
						IsLoopControlVariable(load.Identifier, loopBegin))
						continue;
					// Hoist: remove from loop body, insert just before LoopBegin
					result.RemoveAt(bodyIndex);
					result.Insert(index, load);
					// LoopEndInstruction.Begin still points to the same LoopBeginInstruction object;
					// InstructionIndex is re-set dynamically in VirtualMachine.RunInstructions, so no
					// fixup needed here.
					changed = true;
					break;
				}
				if (changed)
					break;
			}
		}
		return result;
	}

	private static int FindLoopEnd(List<Instruction> instructions, int loopBeginIndex)
	{
		var depth = 0;
		for (var index = loopBeginIndex; index < instructions.Count; index++)
		{
			if (instructions[index] is LoopBeginInstruction)
				depth++;
			else if (instructions[index] is LoopEndInstruction)
			{
				depth--;
				if (depth == 0)
					return index;
			}
		}
		return -1;
	}

	private static HashSet<string> CollectWrittenVariables(List<Instruction> instructions,
		int start, int end)
	{
		var written = new HashSet<string>(StringComparer.Ordinal);
		for (var index = start; index < end; index++)
		{
			switch (instructions[index])
			{
			case StoreFromRegisterInstruction store:
				written.Add(store.Identifier);
				break;
			case StoreVariableInstruction storeVar:
				written.Add(storeVar.Identifier);
				break;
			}
		}
		return written;
	}

	private static bool IsLoopControlVariable(string name, LoopBeginInstruction loopBegin) =>
		name.Equals(Type.ValueLowercase, StringComparison.OrdinalIgnoreCase) ||
		name.Equals(Type.IndexLowercase, StringComparison.OrdinalIgnoreCase) ||
		loopBegin.CustomVariableNames.Any(customVariableName =>
			customVariableName.Equals(name, StringComparison.OrdinalIgnoreCase));
}