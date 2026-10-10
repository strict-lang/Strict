using Strict.Bytecode;
using Strict.Bytecode.Instructions;

namespace Strict.Optimizers;

/// <summary>
/// Removes stores to variables that are never loaded again. A StoreVariableInstruction is dead
/// if the variable identifier never appears in any LoadVariableToRegister or Invoke expression.
/// Also removes dead StoreFromRegisterInstruction chains (e.g. test-helper constants whose
/// values are only used in IS-expressions that are skipped at binary generation).
/// Member variables are always kept as they may be accessed externally.
/// </summary>
public sealed class DeadStoreEliminator : InstructionOptimizer
{
	public override List<Instruction> Optimize(List<Instruction> instructions)
	{
		var loadedVariables = CollectLoadedVariables(instructions);
		instructions.RemoveAll(instruction =>
			instruction is StoreVariableInstruction { IsMember: false } store &&
			!IsElementWrite(store.Identifier) && !loadedVariables.Contains(store.Identifier));
		RemoveDeadRegisterStoreChains(instructions, loadedVariables);
		return instructions;
	}

	/// <summary>
	/// Removes StoreFromRegisterInstruction for variables never loaded, and removes the
	/// immediately preceding single instruction that wrote to that register if the register
	/// has no other consumers.
	/// </summary>
	private static void RemoveDeadRegisterStoreChains(List<Instruction> instructions,
		HashSet<string> loadedVariables)
	{
		bool changed;
		do
		{
			changed = false;
			for (var storeIndex = 0; storeIndex < instructions.Count; storeIndex++)
			{
				if (instructions[storeIndex] is not StoreFromRegisterInstruction store)
					continue;
				if (IsElementWrite(store.Identifier))
					continue;
				if (loadedVariables.Contains(store.Identifier))
					continue;
				var storeRegister = store.Register;
				if (!IsRegisterOnlyUsedByStore(instructions, storeIndex, storeRegister))
					continue;
				var producerIndex = FindProducerInstruction(instructions, storeIndex, storeRegister);
				instructions.RemoveAt(storeIndex);
				if (producerIndex >= 0)
					instructions.RemoveAt(producerIndex < storeIndex
						? producerIndex
						: producerIndex - 1);
				changed = true;
				break;
			}
		} while (changed);
	}

	private static bool IsRegisterOnlyUsedByStore(List<Instruction> instructions, int storeIndex,
		Register storeRegister)
	{
		for (var index = 0; index < instructions.Count; index++)
		{
			if (index == storeIndex)
				continue;
			if (ConsumesRegister(instructions[index], storeRegister))
				return false;
		}
		return true;
	}

	/// <summary>
	/// Returns true when <paramref name="instruction"/> READS from <paramref name="register"/>
	/// (i.e., uses it as an input). Does NOT flag instructions that only WRITE to the register.
	/// </summary>
	private static bool ConsumesRegister(Instruction instruction, Register register) =>
		instruction switch
		{
			BinaryInstruction bin => bin.Registers.Length >= 2 &&
				(bin.Registers[0] == register || bin.Registers[1] == register),
			ListCallInstruction listCall => listCall.IndexValueRegister == register,
			StoreFromRegisterInstruction store => store.Register == register,
			ReturnInstruction ret => ret.Register == register,
			PrintInstruction print => print.ValueRegister == register,
			WriteToListInstruction writeList => writeList.Register == register,
			FieldLoadInstruction fieldLoad => fieldLoad.ObjectRegister == register,
			ConstructValueTypeInstruction construct => construct.FieldRegisters.Contains(register),
			WriteToTableInstruction writeTable => writeTable.Register == register ||
				writeTable.Value == register,
			LoopBeginInstruction loop => loop.Register == register || loop.EndIndex == register,
			_ => false
		};

	private static int FindProducerInstruction(List<Instruction> instructions, int storeIndex,
		Register targetRegister)
	{
		for (var index = storeIndex - 1; index >= 0; index--)
		{
			if (instructions[index] is RegisterInstruction reg && reg.Register == targetRegister)
				return index;
			if (instructions[index] is BinaryInstruction bin && bin.Registers.Length >= 3 &&
				bin.Registers[2] == targetRegister)
				return index;
		}
		return -1;
	}

	private static string? GetReadIdentifier(Instruction instruction) =>
		instruction switch
		{
			LoadVariableToRegister load => load.Identifier,
			ListCallInstruction listCall => listCall.Identifier,
			WriteToListInstruction writeToList => writeToList.Identifier,
			WriteToTableInstruction writeToTable => writeToTable.Identifier,
			RemoveInstruction remove => remove.Identifier,
			StoreVariableInstruction store when IsElementWrite(store.Identifier) => store.Identifier,
			StoreFromRegisterInstruction store when IsElementWrite(store.Identifier) =>
				store.Identifier,
			_ => null
		};

	/// <summary>
	/// List-element writes like image.Colors(colorIndex) = … change the list in place, they are
	/// never dead and read both the list and the index variable.
	/// </summary>
	private static bool IsElementWrite(string identifier) => identifier.Contains('(');

	private static HashSet<string> CollectLoadedVariables(List<Instruction> instructions)
	{
		var loaded = new HashSet<string>(StringComparer.Ordinal);
		foreach (var instruction in instructions)
			if (GetReadIdentifier(instruction) is { } identifier)
			{
				loaded.Add(identifier);
				loaded.Add(identifier.Split('.', '(')[0]);
				var openParenthesis = identifier.LastIndexOf('(');
				if (openParenthesis > 0 && identifier.EndsWith(')'))
					loaded.Add(identifier[(openParenthesis + 1)..^1].Split('.')[0]);
			}
		return loaded;
	}
}