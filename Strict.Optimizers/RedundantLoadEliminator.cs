using Strict.Bytecode;
using Strict.Bytecode.Instructions;

namespace Strict.Optimizers;

/// <summary>
/// Eliminates redundant loads of the same variable within a basic block. When a variable is
/// loaded into a register and then loaded again without any intervening store to that variable
/// (or loop/jump boundaries), the second load is removed and all references to its register
/// are rewritten to use the first load's register.
/// </summary>
public sealed class RedundantLoadEliminator : InstructionOptimizer
{
	public override List<Instruction> Optimize(List<Instruction> instructions)
	{
		var variableToRegister = new Dictionary<string, Register>();
		var registerRemapping = new Dictionary<Register, Register>();
		var toRemove = new List<int>();
		for (var i = 0; i < instructions.Count; i++)
		{
			if (IsBlockBoundary(instructions[i]))
			{
				variableToRegister.Clear();
				continue;
			}
			switch (instructions[i].InstructionType)
			{
			case InstructionType.StoreRegisterToVariable:
				var storeReg = (StoreFromRegisterInstruction)instructions[i];
				variableToRegister.Remove(storeReg.Identifier);
				continue;
			case InstructionType.LoadVariableToRegister:
				var load = (LoadVariableToRegister)instructions[i];
				if (variableToRegister.TryGetValue(load.Identifier, out var existingRegister))
				{
					registerRemapping[load.Register] = existingRegister;
					toRemove.Add(i);
				}
				else
					variableToRegister[load.Identifier] = load.Register;
				break;
			}
		}
		for (var i = toRemove.Count - 1; i >= 0; i--)
			instructions.RemoveAt(toRemove[i]);
		if (registerRemapping.Count > 0)
			RemapRegisters(instructions, registerRemapping);
		return instructions;
	}

	private static bool IsBlockBoundary(Instruction instruction) =>
		instruction.InstructionType is InstructionType.Invoke or >= InstructionType.LoopBegin;

	private static void RemapRegisters(List<Instruction> instructions,
		Dictionary<Register, Register> remapping)
	{
		for (var i = 0; i < instructions.Count; i++)
			if (instructions[i] is BinaryInstruction binary)
			{
				var remapped = false;
				var newRegisters = new Register[binary.Registers.Length];
				for (var r = 0; r < binary.Registers.Length; r++)
				{
					newRegisters[r] = remapping.TryGetValue(binary.Registers[r], out var mapped)
						? mapped
						: binary.Registers[r];
					if (newRegisters[r] != binary.Registers[r])
						remapped = true;
				}
				if (remapped)
					instructions[i] = new BinaryInstruction(binary.InstructionType, newRegisters);
			}
			else if (instructions[i] is StoreFromRegisterInstruction store &&
				remapping.TryGetValue(store.Register, out var mappedStore))
				instructions[i] = new StoreFromRegisterInstruction(mappedStore, store.Identifier);
			else if (instructions[i] is ReturnInstruction ret &&
				remapping.TryGetValue(ret.Register, out var mappedReturn))
				instructions[i] = new ReturnInstruction(mappedReturn); //ncrunch: no coverage
	}
}