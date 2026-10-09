using Strict.Bytecode;
using Strict.Bytecode.Instructions;

namespace Strict.Optimizers;

/// <summary>
/// Eliminates redundant loads of the same variable within a basic block. The repeated load is
/// removed and its reads use the first load's register, but only until its register is written
/// again: registers are reused, a method-wide remap would corrupt unrelated later values.
/// </summary>
public sealed class RedundantLoadEliminator : InstructionOptimizer
{
	public override List<Instruction> Optimize(List<Instruction> instructions)
	{
		var variableToRegister = new Dictionary<string, Register>();
		for (var index = 0; index < instructions.Count; index++)
		{
			var instruction = instructions[index];
			if (IsBlockBoundary(instruction))
			{
				variableToRegister.Clear();
				continue;
			}
			if (instruction is LoadVariableToRegister load &&
				variableToRegister.TryGetValue(load.Identifier, out var existingRegister) &&
				TryRemapReadsUntilOverwritten(instructions, index + 1, load.Register, existingRegister))
			{
				instructions.RemoveAt(index--);
				continue;
			}
			if (instruction is StoreFromRegisterInstruction store)
				variableToRegister.Remove(store.Identifier);
			if (GetWrittenRegister(instruction) is { } written)
				foreach (var staleName in variableToRegister.Where(pair => pair.Value == written).
					Select(pair => pair.Key).ToList())
					variableToRegister.Remove(staleName);
			if (instruction is LoadVariableToRegister newLoad)
				variableToRegister[newLoad.Identifier] = newLoad.Register;
		}
		return instructions;
	}

	private static bool IsBlockBoundary(Instruction instruction) =>
		instruction.InstructionType is InstructionType.Invoke or >= InstructionType.LoopBegin;

	private static bool IsControlFlow(Instruction instruction) =>
		instruction.InstructionType is >= InstructionType.LoopBegin and
			<= InstructionType.JumpToIdIfTrue;

	private static bool TryRemapReadsUntilOverwritten(List<Instruction> instructions, int start,
		Register from, Register to)
	{
		var readPositions = new List<int>();
		var isSourceOverwritten = false;
		var isAfterControlFlow = false;
		for (var index = start; index < instructions.Count; index++)
		{
			var instruction = instructions[index];
			if (GetReadRegisters(instruction).Contains(from))
			{
				if (isSourceOverwritten || isAfterControlFlow || !CanRemap(instruction))
					return false;
				readPositions.Add(index);
			}
			var written = GetWrittenRegister(instruction);
			if (written == from)
				break;
			isSourceOverwritten |= written == to;
			isAfterControlFlow |= IsControlFlow(instruction);
		}
		foreach (var position in readPositions)
			instructions[position] = RemapReads(instructions[position], from, to);
		return true;
	}

	private static IEnumerable<Register> GetReadRegisters(Instruction instruction) =>
		instruction switch
		{
			BinaryInstruction binary => binary.Registers.Length >= 3
				? binary.Registers[..^1]
				: binary.Registers,
			Invoke invoke => invoke.MethodInfo.InstanceRegister is { } instance
				? [.. invoke.MethodInfo.ArgumentRegisters, instance]
				: invoke.MethodInfo.ArgumentRegisters,
			FieldLoadInstruction fieldLoad => [fieldLoad.ObjectRegister],
			ConstructValueTypeInstruction construct => construct.FieldRegisters,
			ListCallInstruction listCall => [listCall.IndexValueRegister],
			WriteToTableInstruction writeToTable => [writeToTable.Register, writeToTable.Value],
			PrintInstruction { ValueRegister: { } printed } => [printed],
			JumpIfNotZero jumpIfNotZero => [jumpIfNotZero.Register],
			StoreFromRegisterInstruction or ReturnInstruction or WriteToListInstruction
				or RemoveInstruction or LoopBeginInstruction => [((RegisterInstruction)instruction).Register],
			_ => []
		};

	private static Register? GetWrittenRegister(Instruction instruction) =>
		instruction switch
		{
			BinaryInstruction { Registers.Length: >= 3 } binary => binary.Registers[^1],
			LoadVariableToRegister or LoadConstantInstruction or SetInstruction or Invoke
				or FieldLoadInstruction or ConstructValueTypeInstruction or ListCallInstruction =>
				((RegisterInstruction)instruction).Register,
			_ => null
		};

	private static bool CanRemap(Instruction instruction) =>
		instruction is BinaryInstruction or Invoke or FieldLoadInstruction
			or ConstructValueTypeInstruction or StoreFromRegisterInstruction or ReturnInstruction
			or WriteToListInstruction;

	private static Instruction RemapReads(Instruction instruction, Register from, Register to)
	{
		Register Map(Register register) =>
			register == from
				? to
				: register;
		return instruction switch
		{
			BinaryInstruction binary => new BinaryInstruction(binary.InstructionType, binary.Registers.
				Select((register, index) => index < 2
					? Map(register)
					: register).ToArray()),
			Invoke invoke => new Invoke(invoke.Register, new InvokeMethodInfo(
				invoke.MethodInfo.TypeFullName, invoke.MethodInfo.MethodName,
				invoke.MethodInfo.ParameterNames, invoke.MethodInfo.ReturnTypeName,
				invoke.MethodInfo.ArgumentRegisters.Select(Map).ToArray(),
				invoke.MethodInfo.InstanceRegister is { } instance
					? Map(instance)
					: null)),
			FieldLoadInstruction fieldLoad => new FieldLoadInstruction(fieldLoad.Register, to,
				fieldLoad.FieldName),
			ConstructValueTypeInstruction construct => new ConstructValueTypeInstruction(
				construct.Register, construct.ReturnType, construct.FieldRegisters.Select(Map).ToArray()),
			StoreFromRegisterInstruction store => new StoreFromRegisterInstruction(to, store.Identifier),
			WriteToListInstruction writeToList => new WriteToListInstruction(to, writeToList.Identifier),
			_ => new ReturnInstruction(to)
		};
	}
}
