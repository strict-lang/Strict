using Strict.Runtime.Instructions;

namespace Strict.Optimizers;

/// <summary>
/// Removes stores to variables that are never loaded again. A StoreVariableStatement is dead
/// if the variable identifier never appears in any LoadVariableToRegister or
/// StoreFromRegisterStatement in the rest of the instruction list. Member variables are always
/// kept as they may be accessed externally.
/// </summary>
public sealed class DeadStoreEliminator : InstructionOptimizer
{
	public override List<Instruction> Optimize(List<Instruction> instructions)
	{
		var usedVariables = CollectUsedVariables(instructions);
		instructions.RemoveAll(statement =>
			statement is StoreVariableInstruction store && !store.IsMember &&
			!usedVariables.Contains(store.Identifier));
		return instructions;
	}

	private static HashSet<string> CollectUsedVariables(List<Instruction> instructions)
	{
		var used = new HashSet<string>();
		foreach (var instruction in instructions)
			if (instruction is LoadVariableToRegister load)
				used.Add(load.Identifier);
			else if (instruction is StoreFromRegisterInstruction store)
				used.Add(store.Identifier);
		return used;
	}
}