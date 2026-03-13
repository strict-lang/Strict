using Strict.Bytecode.Instructions;
using Strict.Expressions;

namespace Strict.Optimizers;

/// <summary>
/// Removes stores to variables that are never loaded again. A StoreVariableInstruction is dead
/// if the variable identifier never appears in any LoadVariableToRegister,
/// StoreFromRegisterInstruction, or as the instance of an Invoke instruction.
/// Member variables are always kept as they may be accessed externally.
/// </summary>
public sealed class DeadStoreEliminator : InstructionOptimizer
{
	public override List<Instruction> Optimize(List<Instruction> instructions)
	{
		var usedVariables = CollectUsedVariables(instructions);
		instructions.RemoveAll(instruction =>
			instruction is StoreVariableInstruction { IsMember: false } store &&
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
			else if (instruction is Invoke { Method: { Instance: VariableCall instanceVar } })
				used.Add(instanceVar.Variable.Name);
		return used;
	}
}