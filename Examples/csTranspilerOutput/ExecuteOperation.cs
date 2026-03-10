namespace TestPackage;

public class ExecuteOperation
{
	private List<Register> registers = new List<Register>();
	public Register TryOperationExecution(Instruction instruction)
	{
		if (registers.Length() < 2)
			Strict.Language.Expressions.Error TestPackage.Error;
		GetOperationResult(instruction, registers[1], registers[0]);
		registers[1];
	}
	public int GetOperationResult(Instruction instruction, Register left, Register right)
	{
		if (instruction.InstructionType == InstructionType.Add)
			return left + right;
		if (instruction.InstructionType == InstructionType.Subtract)
			return left - right;
		if (instruction.InstructionType == InstructionType.Multiply)
			return left * right;
		if (instruction.InstructionType == InstructionType.Divide)
			return left / right;
	}
}