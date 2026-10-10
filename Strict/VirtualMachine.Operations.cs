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
			_ => new ValueInstance(GetNumberResultType(right), CalculateNumber(
				instruction.InstructionType, left.GetArithmeticNumber(), right.GetArithmeticNumber()))
		};
	}

	/// <summary>
	/// Number wrappers like Degrees (e.g. the implicit value) calculate with their number member.
	/// </summary>
	private double CalculateNumber(InstructionType operation, double left, double right) =>
		operation switch
		{
			InstructionType.Multiply => left * right,
			InstructionType.Divide => left / right,
			InstructionType.Modulo => left % right,
			InstructionType.Power => Math.Pow(left, right),
			_ => throw Fail("Unsupported binary operation: " + operation) //ncrunch: no coverage
		};

	/// <summary>
	/// Calculating with a number wrapper like Degrees gives a plain Number, like the interpreter.
	/// </summary>
	private Type GetNumberResultType(ValueInstance operand) =>
		operand.IsFlatNumeric || operand.TryGetValueTypeInstance() != null
			? executable.numberType
			: operand.GetType();

	private ValueInstance AddValueInstances(ValueInstance left, ValueInstance right)
	{
		if (left.IsList)
			return new ValueInstance(left.List.Appended(right.IsList &&
				!left.List.ReturnType.GetFirstImplementation().IsList
					? right.List.Items
					: [right]));
		if (right.IsList)
			return new ValueInstance(right.List.ReturnType, [left, .. right.List.Items]);
		if (left.IsText || right.IsText)
			return new ValueInstance(ConvertToText(left).Text + ConvertToText(right).Text);
		return new ValueInstance(GetNumberResultType(right),
			left.GetArithmeticNumber() + right.GetArithmeticNumber());
	}

	private ValueInstance SubtractValueInstances(ValueInstance left, ValueInstance right)
	{
		if (left.IsList)
			return SubtractFromList(left, right);
		if (left.IsText || right.IsText)
			throw Fail("Text subtraction is not supported: '" + left + "' - '" + right +
				"'"); //ncrunch: no coverage
		return new ValueInstance(GetNumberResultType(left),
			left.GetArithmeticNumber() - right.GetArithmeticNumber());
	}

	/// <summary>
	/// Own method, the RemoveAll lambda would otherwise allocate its closure on every subtraction.
	/// </summary>
	private static ValueInstance SubtractFromList(ValueInstance left, ValueInstance right)
	{
		var items = new List<ValueInstance>(left.List.Items);
		if (right.IsList && !left.List.ReturnType.GetFirstImplementation().IsList)
			foreach (var item in right.List.Items)
				items.Remove(item);
		else
			items.RemoveAll(item => item.Equals(right));
		return new ValueInstance(left.List.ReturnType, items.ToArray());
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
			InstructionType.GreaterThan => left.GetArithmeticNumber() > right.GetArithmeticNumber(),
			InstructionType.LessThan => left.GetArithmeticNumber() < right.GetArithmeticNumber(),
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
