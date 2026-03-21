namespace TestPackage;

public class ExecuteOperation
{
	private List<Register> registers = new List<Register>();
	public Register TryOperationExecution(Instruction instruction)
	{
		if (registers.Length() < 2)
			new Error("Text: \"OperandsRequired\"", new List(Stacktrace)(new Stacktrace(new Method("TryOperationExecution", new Type("ExecuteOperation", "TestPackage")), "C:\\code\\GitHub\\strict-lang\\Strict\\ExecuteOperation.strict", 2)));
		GetOperationResult(instruction, registers[1], registers[0]);
		registers[1];
	}
	public int GetOperationResult(Instruction instruction, Register left, Register right)
	{
	switch (instruction.InstructionType)
	{
		case InstructionType.Add: return left + right;
		case InstructionType.Subtract: return left - right;
		case InstructionType.Multiply: return left * right;
		case InstructionType.Divide: return left / right;
	}
	}
}