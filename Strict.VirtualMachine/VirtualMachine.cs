using Strict.Language;

namespace Strict.VirtualMachine;

public sealed class VirtualMachine
{
	public VirtualMachine Execute(IReadOnlyList<Statement> statements)
	{
		for (instructionIndex = 0; instructionIndex != -1 && instructionIndex < statements.Count;
			instructionIndex++)
			ExecuteStatement(statements[instructionIndex]);
		return this;
	}

	private void ExecuteStatement(Statement statement)
	{
		if (statement.Instruction == Instruction.Return)
		{
			instructionIndex = -2;
			Returns = Registers[statement.Registers[0]];
			return;
		}
		TryStoreInstructions(statement);
		TryLoadInstructions(statement);
		TryExecute(statement);
	}

	private void TryStoreInstructions(Statement statement)
	{
		if (statement.Instance == null)
			return;
		if (statement.Instruction == Instruction.Set)
			foreach (var register in statement.Registers)
				Registers[register] = statement.Instance;
		else if (statement.Instruction == Instruction.StoreVariable)
			variables[((StoreStatement)statement).Identifier] = statement.Instance;
	}

	private void TryLoadInstructions(Statement statement)
	{
		if (statement.Instruction is Instruction.Load)
			Registers[statement.Registers[0]] =
				variables[((LoadVariableStatement)statement).Identifier];
		else if (statement is LoadConstantStatement loadConstantStatement)
			Registers[loadConstantStatement.Register] = loadConstantStatement.ConstantInstance;
	}

	private void TryExecute(Statement statement)
	{
		var instructionPosition = (int)statement.Instruction;
		if (instructionPosition is >= (int)Instruction.SetLoadSeparator
			and < (int)Instruction.BinaryOperatorsSeparator)
			TryBinaryOperationExecution(statement);
		else if (instructionPosition is >= 100 and < (int)Instruction.ConditionalSeparator)
			TryConditionalOperationExecution(statement);
		else if (instructionPosition is >= 9 and <= (int)Instruction.JumpsSeparator)
			TryJumpOperation((JumpStatement)statement);
	}

	public Dictionary<Register, Instance> Registers { get; } = new();
	private readonly Dictionary<string, Instance> variables = new();
	private int instructionIndex;
	public Instance? Returns { get; private set; }

	private void TryBinaryOperationExecution(Statement statement)
	{
		var (right, left) = GetOperands(statement);
		Registers[statement.Registers[^1]] = statement.Instruction switch
		{
			Instruction.Add => GetAdditionResult(left, right),
			Instruction.Subtract => new Instance(right.ReturnType,
				Convert.ToDouble(left.Value) - Convert.ToDouble(right.Value)),
			Instruction.Multiply => new Instance(right.ReturnType,
				Convert.ToDouble(left.Value) * Convert.ToDouble(right.Value)),
			Instruction.Divide => new Instance(right.ReturnType,
				Convert.ToDouble(left.Value) / Convert.ToDouble(right.Value)),
			_ => Registers[statement.Registers[^1]] //ncrunch: no coverage
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
		Registers.Count < 2
			? throw new OperandsRequired()
			: (Registers[statement.Registers[1]], Registers[statement.Registers[0]]);

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

	private void TryJumpOperation(JumpStatement statement)
	{
		if (conditionFlag && statement.Instruction is Instruction.JumpIfTrue ||
			!conditionFlag && statement.Instruction is Instruction.JumpIfFalse ||
			statement.Instruction is Instruction.JumpIfNotZero &&
			Convert.ToInt32(Registers[statement.RegisterToCheckForZero].Value) != 0)
			instructionIndex += Convert.ToInt32(statement.Steps);
	}

	public class OperandsRequired : Exception { }
}