namespace Strict.VirtualMachine;

public sealed class VirtualMachine
{
	public Dictionary<Register, double> Execute(IReadOnlyList<Statement> statements)
	{
		for (instructionIndex = 0; instructionIndex < statements.Count; instructionIndex++)
			ExecuteStatement(statements[instructionIndex]);
		return registers;
	}

	private void ExecuteStatement(Statement statement)
	{
		if (statement.Instruction == Instruction.Set)
			foreach (var register in statement.Registers)
				registers[register] = statement.Value;
		else
			TryExecute(statement);
	}

	private void TryExecute(Statement statement)
	{
		var instructionPosition = (int)statement.Instruction;
		if (instructionPosition is >= 1 and < (int)Instruction.BinaryOperatorsSeparator)
			TryOperationExecution(statement);
		else if (instructionPosition is >= 100 and < (int)Instruction.ConditionalSeparator)
			TryConditionalOperationExecution(statement);
		else if (instructionPosition is >= 9 and <= (int)Instruction.JumpsSeparator)
			TryJumpOperation(statement);
	}

	private readonly Dictionary<Register, double> registers = new();
	private int instructionIndex;

	private void TryOperationExecution(Statement statement)
	{
		var (right, left) = GetOperands(statement);
		registers[statement.Registers[^1]] = statement.Instruction switch
		{
			Instruction.Add => left + right,
			Instruction.Subtract => left - right,
			Instruction.Multiply => left * right,
			Instruction.Divide => left / right,
			_ => registers[statement.Registers[^1]] //ncrunch: no coverage
		};
	}

	private (double, double) GetOperands(Statement statement) =>
		(registers[statement.Registers[1]], registers[statement.Registers[0]]);

	private void TryConditionalOperationExecution(Statement statement)
	{
		var (right, left) = GetOperands(statement);
		var result = statement.Instruction switch
		{
			Instruction.GreaterThan => left > right,
			Instruction.LessThan => left < right,
			Instruction.Equal => left == right,
			Instruction.NotEqual => left != right,
			_ => false //ncrunch: no coverage
		};
		conditionFlag = result;
	}

	private bool conditionFlag;

	private void TryJumpOperation(Statement statement)
	{
		if (statement.Instruction == Instruction.JumpIfTrue && conditionFlag)
			instructionIndex += (int)statement.Value;
		else if (statement.Instruction == Instruction.JumpIfFalse && !conditionFlag)
			instructionIndex += (int)statement.Value;
		else if (statement.Instruction == Instruction.JumpIfNotZero && registers[statement.Registers[0]] != 0)
			instructionIndex += (int)statement.Value;
	}
}