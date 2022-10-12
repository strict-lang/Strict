namespace Strict.VirtualMachine;

public sealed class VirtualMachine
{
	public double Execute(IReadOnlyList<Statement> statements)
	{
		for (var instructionIndex = 0; instructionIndex < statements.Count; instructionIndex++)
			ExecuteStatement(statements[instructionIndex], ref instructionIndex);
		return stack.Pop();
	}

	private void ExecuteStatement(Statement statement, ref int instructionIndex)
	{
		if (statement.Instruction == Instruction.Set)
			registers[statement.Register] = statement.Value;
		else if (statement.Instruction == Instruction.JumpIfNotZero)
		{
			if (registers[Register.A] != 0)
				instructionIndex -= (int)statement.Value;
		}
		else if (statement.Instruction == Instruction.Push)
			stack.Push(statement.Value);
		else
			ExecuteOperation(statement);
	}

	private readonly Stack<double> stack = new();
	private readonly Dictionary<Register, double> registers = new();

	private void ExecuteOperation(Statement statement)
	{
		var right = stack.Pop();
		var left = statement.Register != Register.None
			? registers[statement.Register]
			: stack.Pop();
		var result = statement.Instruction switch
		{
			Instruction.Add => left + right,
			Instruction.Subtract => left - right,
			Instruction.Multiply => left * right,
			Instruction.Divide => left / right,
			_ => throw new NotSupportedException() //ncrunch: no coverage
		};
		if (statement.Register != Register.None)
			registers[statement.Register] = result;
		else
			stack.Push(result);
	}
}