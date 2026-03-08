using Strict.Runtime;
using Strict.Runtime.Statements;

namespace Strict.Optimizers;

/// <summary>
/// Eliminates redundant loads of the same variable within a basic block. When a variable is
/// loaded into a register and then loaded again without any intervening store to that variable
/// (or loop/jump boundaries), the second load is removed and all references to its register
/// are rewritten to use the first load's register.
/// </summary>
public sealed class RedundantLoadEliminator : StatementOptimizer
{
	public override List<Statement> Optimize(List<Statement> statements)
	{
		var variableToRegister = new Dictionary<string, Register>();
		var registerRemapping = new Dictionary<Register, Register>();
		var toRemove = new List<int>();
		for (var i = 0; i < statements.Count; i++)
		{
			if (IsBlockBoundary(statements[i]))
			{
				variableToRegister.Clear();
				continue;
			}
			if (statements[i] is StoreFromRegisterStatement storeReg)
			{
				variableToRegister.Remove(storeReg.Identifier);
				continue;
			}
			if (statements[i] is not LoadVariableToRegister load)
				continue;
			if (variableToRegister.TryGetValue(load.Identifier, out var existingRegister))
			{
				registerRemapping[load.Register] = existingRegister;
				toRemove.Add(i);
			}
			else
				variableToRegister[load.Identifier] = load.Register;
		}
		for (var i = toRemove.Count - 1; i >= 0; i--)
			statements.RemoveAt(toRemove[i]);
		if (registerRemapping.Count > 0)
			RemapRegisters(statements, registerRemapping);
		return statements;
	}

	private static bool IsBlockBoundary(Statement statement) =>
		statement.Instruction is Instruction.LoopBegin or Instruction.LoopEnd or
			Instruction.Jump or Instruction.JumpIfTrue or Instruction.JumpIfFalse or
			Instruction.JumpIfNotZero or Instruction.JumpEnd or Instruction.JumpToIdIfFalse or
			Instruction.JumpToIdIfTrue;

	private static void RemapRegisters(List<Statement> statements,
		Dictionary<Register, Register> remapping)
	{
		for (var i = 0; i < statements.Count; i++)
			if (statements[i] is Binary binary)
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
					statements[i] = new Binary(binary.Instruction, newRegisters);
			}
	}
}