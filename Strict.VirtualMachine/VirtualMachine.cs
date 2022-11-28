using Strict.Language;
using Strict.Language.Expressions;

namespace Strict.VirtualMachine;

public sealed class VirtualMachine
{
	private readonly Dictionary<string, Instance> variables = new();
	private bool conditionFlag;
	private int instructionIndex;
	private IList<Statement> statements = new List<Statement>();
	public Dictionary<Register, Instance> Registers { get; } = new();
	public Instance? Returns { get; private set; }

	public VirtualMachine Execute(IList<Statement> allStatements)
	{
		Clear();
		statements = allStatements;
		for (instructionIndex = 0; instructionIndex != -1 && instructionIndex < allStatements.Count;
			instructionIndex++)
			ExecuteStatement(allStatements[instructionIndex]);
		return this;
	}

	private void Clear()
	{
		variables.Clear();
		conditionFlag = false;
		instructionIndex = 0;
		statements.Clear();
		Registers.Clear();
		Returns = null;
	}

	private void ExecuteStatement(Statement statement)
	{
		if (TryExecuteReturn(statement))
			return;
		TryStoreInstructions(statement);
		TryLoadInstructions(statement);
		TryLoopInitInstruction(statement);
		TryExecute(statement);
	}

	private bool TryExecuteReturn(Statement statement)
	{
		if (statement.Instruction != Instruction.Return)
			return false;
		Returns = Registers[statement.Registers[0]];
		if (!Returns.Value.GetType().IsPrimitive && Returns.Value is not Value)
			return false;
		instructionIndex = -2;
		return true;
	}

	private void TryLoopInitInstruction(Statement statement)
	{
		if (statement is InitLoopStatement initLoopStatement)
		{
			if (variables.ContainsKey("index"))
				variables["index"].Value = Convert.ToInt32(variables["index"].Value) + 1;
			else
				variables.Add("index", new Instance(0));
			variables.TryGetValue(initLoopStatement.Identifier, out var iterableVariable);
			if (iterableVariable != null)
				AlterValueVariable(iterableVariable);
		}
	}

	private void AlterValueVariable(Instance iterableVariable)
	{
		var index = Convert.ToInt32(variables["index"].Value);
		var value = iterableVariable.Value.ToString();
		if (iterableVariable.ReturnType is { IsList: true })
			variables["value"] = new Instance(((List)iterableVariable.Value).Values[index]);
		else if (iterableVariable.ReturnType?.Name == Base.Number)
			variables["value"] = new Instance(Convert.ToInt32(iterableVariable.Value) + index);
		else if (iterableVariable.ReturnType?.Name == Base.Text && value != null)
			variables["value"] = new Instance(value[index].ToString());
	}

	private void TryStoreInstructions(Statement statement)
	{
		if (statement.Instruction > Instruction.SetLoadSeparator)
			return;
		if (statement.Instruction == Instruction.Set && statement.Instance != null)
			foreach (var register in statement.Registers)
				Registers[register] = statement.Instance;
		else if (statement.Instruction == Instruction.StoreVariable && statement.Instance != null)
			variables[((StoreStatement)statement).Identifier] = statement.Instance;
		else if (statement.Instruction == Instruction.StoreFromRegister)
			variables[((StoreFromRegisterStatement)statement).Identifier] =
				Registers[statement.Registers[0]];
	}

	private void TryLoadInstructions(Statement statement)
	{
		if (statement.Instruction is Instruction.Load)
			LoadVariableIntoRegister((LoadVariableStatement)statement);
		else if (statement is LoadConstantStatement loadConstantStatement)
			Registers[loadConstantStatement.Register] = loadConstantStatement.ConstantInstance;
	}

	private void LoadVariableIntoRegister(LoadVariableStatement statement) => Registers[statement.Register] = variables[statement.Identifier];

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
		if (left.ReturnType?.Name == Base.Number && right.ReturnType?.Name == Base.Number)
			return new Instance(right.ReturnType,
				Convert.ToDouble(left.Value) + Convert.ToDouble(right.Value));
		if (left.ReturnType?.Name == Base.Text && right.ReturnType?.Name == Base.Text)
			return new Instance(right.ReturnType, left.Value.ToString() + right.Value);
		if (right.ReturnType?.Name == Base.Text && left.ReturnType?.Name == Base.Number)
			return new Instance(right.ReturnType, left.Value.ToString() + right.Value);
		return new Instance(right.ReturnType, left.Value.ToString() + right.Value);
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

	private void TryJumpOperation(JumpStatement statement)
	{
		if (conditionFlag && statement.Instruction is Instruction.JumpIfTrue ||
			!conditionFlag && statement.Instruction is Instruction.JumpIfFalse ||
			statement.Instruction is Instruction.JumpIfNotZero &&
			Convert.ToInt32(Registers[statement.RegisterToCheckForZero].Value) != 0)
			instructionIndex += Convert.ToInt32(statement.Steps);
		else if (!conditionFlag && statement.Instruction is Instruction.JumpToIdIfFalse)
		{
			var id = ((JumpViaIdStatement)statement).Id;
			var endIndex = statements.IndexOf(statements.First(jumpStatement =>
				jumpStatement.Instruction is Instruction.JumpEnd &&
				jumpStatement is JumpViaIdStatement jumpViaId && jumpViaId.Id == id));
			if (endIndex != -1)
				instructionIndex = endIndex;
		}
	}

	public class OperandsRequired : Exception { }
}