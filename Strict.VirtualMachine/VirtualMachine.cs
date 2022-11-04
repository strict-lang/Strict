using Strict.Language;

namespace Strict.VirtualMachine;

public sealed class VirtualMachine
{
	public Dictionary<Register, Instance> Execute(IReadOnlyList<Statement> statements)
	{
		for (instructionIndex = 0; instructionIndex != -1 && instructionIndex < statements.Count; instructionIndex++)
			ExecuteStatement(statements[instructionIndex]);
		return registers;
	}

	private void ExecuteStatement(Statement statement)
	{
		if (statement.Instruction == Instruction.Return)
		{
			instructionIndex = -1;
			return;
		}
		if (statement.Instruction == Instruction.Set && statement.Instance != null)
			foreach (var register in statement.Registers)
				registers[register] = statement.Instance;
		else if (statement.Instruction == Instruction.LoadConstant)
			registers[statement.Registers[0]] = statement.Instance ?? throw new InvalidOperationException();
		else
			TryExecute(statement);
	}

	private void TryExecute(Statement statement)
	{
		var instructionPosition = (int)statement.Instruction;
		if (instructionPosition is >= (int)Instruction.SetLoadSeparator
			and < (int)Instruction.BinaryOperatorsSeparator)
			TryOperationExecution(statement);
		else if (instructionPosition is >= 100 and < (int)Instruction.ConditionalSeparator)
			TryConditionalOperationExecution(statement);
		else if (instructionPosition is >= 9 and <= (int)Instruction.JumpsSeparator)
			TryJumpOperation(statement);
	}

	private readonly Dictionary<Register, Instance> registers = new();
	private int instructionIndex;

	private void TryOperationExecution(Statement statement)
	{
		var (right, left) = GetOperands(statement);
		registers[statement.Registers[^1]] = statement.Instruction switch
		{
			Instruction.Add => GetAdditionResult(left, right),
			Instruction.Subtract => new Instance(right.ReturnType,
				Convert.ToDouble(left.Value) - Convert.ToDouble(right.Value)),
			Instruction.Multiply => new Instance(right.ReturnType,
				Convert.ToDouble(left.Value) * Convert.ToDouble(right.Value)),
			Instruction.Divide => new Instance(right.ReturnType,
				Convert.ToDouble(left.Value) / Convert.ToDouble(right.Value)),
			_ => registers[statement.Registers[^1]] //ncrunch: no coverage
		};
	}

	private static Instance GetAdditionResult(Instance left, Instance right)
	{
		if (left.ReturnType.Name == Base.Number && right.ReturnType.Name == Base.Number)
			return new Instance(right.ReturnType,
				Convert.ToDouble(left.Value) + Convert.ToDouble(right.Value));
		if (left.ReturnType.Name == Base.Text && right.ReturnType.Name == Base.Text)
			return new Instance(right.ReturnType, left.Value.ToString() + right.Value);
		if (right.ReturnType.Name == Base.Text && left.ReturnType.Name == Base.Number)
			return new Instance(right.ReturnType, left.Value.ToString() + right.Value);
		return new Instance(right.ReturnType, right.Value.ToString() + left.Value);
	}

	private (Instance, Instance) GetOperands(Statement statement) =>
		registers.Count < 2
			? throw new OperandsRequired()
			: (registers[statement.Registers[1]], registers[statement.Registers[0]]);

	private void TryConditionalOperationExecution(Statement statement)
	{
		var (right, left) = GetOperands(statement);
		conditionFlag = statement.Instruction switch
		{
			Instruction.GreaterThan => Convert.ToDouble(left.Value) > Convert.ToDouble(right.Value),
			Instruction.LessThan => Convert.ToDouble(left.Value) < Convert.ToDouble(right.Value),
			Instruction.Equal => left.Value.Equals(right.Value),
			Instruction.NotEqual => !left.Value.Equals(right.Value),
			_ => false //ncrunch: no coverage
		};
	}

	private bool conditionFlag;

	private void TryJumpOperation(Statement statement)
	{
		if (statement.Instance == null)
			return;
		if (statement.Instruction == Instruction.JumpIfTrue && conditionFlag)
			instructionIndex += Convert.ToInt32(statement.Instance.Value);
		else if (statement.Instruction == Instruction.JumpIfFalse && !conditionFlag)
			instructionIndex += Convert.ToInt32(statement.Instance.Value);
		else if (statement.Instruction == Instruction.JumpIfNotZero &&
			Convert.ToInt32(registers[statement.Registers[0]].Value) != 0)
			instructionIndex += Convert.ToInt32(statement.Instance.Value);
	}

	public class OperandsRequired : Exception { }
}