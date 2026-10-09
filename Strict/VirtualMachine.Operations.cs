using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Type = Strict.Language.Type;

namespace Strict;

public sealed partial class VirtualMachine
{
	private void ExecuteBinaryInstruction(BinaryInstruction instruction)
	{
		if (instruction.IsConditional())
			ExecuteConditionalOperation(instruction);
		else
			ExecuteBinaryOperation(instruction);
	}

	private void ExecuteBinaryOperation(BinaryInstruction instruction)
	{
		var (right, left) = GetOperands(instruction);
		Memory.Registers[instruction.Registers[^1]] = instruction.InstructionType switch
		{
			InstructionType.Add => AddValueInstances(left, right),
			InstructionType.Subtract => SubtractValueInstances(left, right),
			InstructionType.Multiply => new ValueInstance(right.GetType(), left.Number * right.Number),
			InstructionType.Divide => new ValueInstance(right.GetType(), left.Number / right.Number),
			InstructionType.Modulo => new ValueInstance(right.GetType(), left.Number % right.Number),
			_ => throw Fail("Unsupported binary operation: " +
				instruction.InstructionType) //ncrunch: no coverage
		};
	}

	private static ValueInstance AddValueInstances(ValueInstance left, ValueInstance right)
	{
		if (left.IsList)
		{
			var items = new List<ValueInstance>(left.List.Items);
			if (right.IsList && !left.List.ReturnType.GetFirstImplementation().IsList)
				items.AddRange(right.List.Items);
			else
				items.Add(right);
			return new ValueInstance(left.List.ReturnType, items.ToArray());
		}
		if (right.IsList)
			return new ValueInstance(right.List.ReturnType, [left, .. right.List.Items]);
		if (left.IsText || right.IsText)
			return new ValueInstance(ConvertToText(left).Text + ConvertToText(right).Text);
		return new ValueInstance(right.GetType(), left.Number + right.Number);
	}

	private ValueInstance SubtractValueInstances(ValueInstance left, ValueInstance right)
	{
		if (left.IsList)
		{
			var items = new List<ValueInstance>(left.List.Items);
			var removeIndex = items.FindIndex(item => item.Equals(right));
			if (removeIndex >= 0)
				items.RemoveAt(removeIndex);
			return new ValueInstance(left.List.ReturnType, items.ToArray());
		}
		if (left.IsText || right.IsText)
			throw Fail("Text subtraction is not supported: '" + left + "' - '" + right +
				"'"); //ncrunch: no coverage
		return new ValueInstance(left.GetType(), left.Number - right.Number);
	}

	private (ValueInstance, ValueInstance) GetOperands(BinaryInstruction instruction) =>
		instruction.Registers.Length < 2
			? throw new OperandsRequired()
			: (Memory.Registers[instruction.Registers[1]], Memory.Registers[instruction.Registers[0]]);

	private void ExecuteConditionalOperation(BinaryInstruction instruction)
	{
		var (right, left) = GetOperands(instruction);
		conditionFlag = instruction.InstructionType switch
		{
			InstructionType.GreaterThan => left.Number > right.Number,
			InstructionType.LessThan => left.Number < right.Number,
			InstructionType.Equal => left.Equals(right),
			InstructionType.NotEqual => !left.Equals(right),
			_ => throw Fail("Unsupported conditional operation: " +
				instruction.InstructionType) //ncrunch: no coverage
		};
		// When used as a value expression (not only as if-condition), write a Boolean result.
		if (instruction.Registers.Length >= 3)
			Memory.Registers[instruction.Registers[^1]] =
				new ValueInstance(executable.booleanType, conditionFlag);
	}

	private void TryJumpOperation(Jump instruction)
	{
		if ((conditionFlag && instruction.InstructionType is InstructionType.JumpIfTrue) ||
			(!conditionFlag && instruction.InstructionType is InstructionType.JumpIfFalse))
			instructionIndex += instruction.InstructionsToSkip;
	}

	private void TryJumpIfOperation(JumpIfNotZero instruction)
	{
		if (Memory.Registers[instruction.Register].Number > 0)
			instructionIndex += instruction.InstructionsToSkip;
	}

	private void TryJumpToIdOperation(JumpToId instruction)
	{
		if ((!conditionFlag && instruction.InstructionType is InstructionType.JumpToIdIfFalse) ||
			(conditionFlag && instruction.InstructionType is InstructionType.JumpToIdIfTrue))
		{
			var endIndex = FindJumpEndInstructionIndex(instruction.Id);
			if (endIndex != -1)
				instructionIndex = endIndex;
		}
	}

	private int FindJumpEndInstructionIndex(int id)
	{
		for (var index = 0; index < instructions.Count; index++)
			if (instructions[index].InstructionType == InstructionType.JumpEnd &&
				((JumpToId)instructions[index]).Id == id)
				return index;
		return -1; //ncrunch: no coverage
	}
}
