namespace TestPackage;

public class ExecuteOperation
{
	private List<Register> registers = new List<Register>();
	public Register TryOperationExecution(Statement statement)
	{
		if (registers.Length() < 2)
			Strict.Language.Expressions.Error TestPackage.Error;
		GetOperationResult(statement, registers[1], registers[0]);
		registers[1];
	}
	public int GetOperationResult(Statement statement, Register left, Register right)
	{
		if (statement.Instruction == Instruction.Add)
			return left + right;
		if (statement.Instruction == Instruction.Subtract)
			return left - right;
		if (statement.Instruction == Instruction.Multiply)
			return left * right;
		if (statement.Instruction == Instruction.Divide)
			return left / right;
	}
}