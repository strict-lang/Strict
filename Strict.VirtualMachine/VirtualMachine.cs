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
		TryInvokeInstruction(statement);
		TryExecute(statement);
	}

	private void TryInvokeInstruction(Statement statement)
	{
		if (statement is not InvokeStatement { MethodCall: { } } invokeStatement)
			return;
		var arguments = FormArgumentsForMethodCall(invokeStatement);
		var methodStatements =
			new ByteCodeGenerator(new InvokedMethod(
				((Body)invokeStatement.MethodCall.Method.GetBodyAndParseIfNeeded()).Expressions,
				arguments)).Generate();
		var instance = Execute(methodStatements).Returns;
		if (instance != null)
			Registers[invokeStatement.Register] = instance;
	}

	private Dictionary<string, Instance> FormArgumentsForMethodCall(InvokeStatement invokeStatement)
	{
		var arguments = new Dictionary<string, Instance>();
		if (invokeStatement.MethodCall == null)
			return arguments; // ncrunch: no coverage
		for (var index = 0; index < invokeStatement.MethodCall.Method.Parameters.Count; index++)
		{
			var argument = invokeStatement.MethodCall.Arguments[index];
			var argumentInstance = argument is Value argumentValue
				? new Instance(argumentValue.ReturnType, argumentValue.Data)
				: variables[argument.ToString()];
			arguments.Add(invokeStatement.MethodCall.Method.Parameters[index].Name, argumentInstance);
		}
		return arguments;
	}

	private bool TryExecuteReturn(Statement statement)
	{
		if (statement is not ReturnStatement returnStatement)
			return false;
		Returns = Registers[returnStatement.Register];
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
				variables.Add("index", new Instance(Base.Number, 0));
			variables.TryGetValue(initLoopStatement.Identifier, out var iterableVariable);
			if (iterableVariable != null)
				AlterValueVariable(iterableVariable);
		}
	}

	private void AlterValueVariable(Instance iterableVariable)
	{
		var index = Convert.ToInt32(variables["index"].Value);
		var value = iterableVariable.Value.ToString();
		if (iterableVariable.ReturnType?.Name == Base.Text && value != null)
			variables["value"] = new Instance(Base.Number, value[index].ToString());
		else if (iterableVariable.ReturnType is GenericTypeImplementation genericIterable &&
			genericIterable.Generic.Name == Base.List)
			variables["value"] = new Instance(((List<Expression>)iterableVariable.Value)[index]);
		else if (iterableVariable.ReturnType?.Name == Base.Number)
			variables["value"] =
				new Instance(Base.Number, Convert.ToInt32(iterableVariable.Value) + index);
	}

	private void TryStoreInstructions(Statement statement)
	{
		if (statement.Instruction > Instruction.SetLoadSeparator)
			return;
		if (statement is SetStatement setStatement)
			Registers[setStatement.Register] = setStatement.Instance;
		else if (statement is StoreVariableStatement storeVariableStatement)
			variables[storeVariableStatement.Identifier] = storeVariableStatement.Instance;
		else if (statement is StoreFromRegisterStatement storeFromRegisterStatement)
			variables[storeFromRegisterStatement.Identifier] =
				Registers[storeFromRegisterStatement.Register];
	}

	private void TryLoadInstructions(Statement statement)
	{
		if (statement is LoadVariableStatement loadVariableStatement)
			LoadVariableIntoRegister(loadVariableStatement);
		else if (statement is LoadConstantStatement loadConstantStatement)
			Registers[loadConstantStatement.Register] = loadConstantStatement.Instance;
	}

	private void LoadVariableIntoRegister(LoadVariableStatement statement) =>
		Registers[statement.Register] = variables[statement.Identifier];

	private void TryExecute(Statement statement)
	{
		if (statement is BinaryStatement binaryStatement)
		{
			if (binaryStatement.IsConditional())
				TryConditionalOperationExecution(binaryStatement);
			else
				TryBinaryOperationExecution(binaryStatement);
		}
		else if (statement is JumpStatement jumpStatement)
		{
			TryJumpOperation(jumpStatement);
		}
	}

	private void TryBinaryOperationExecution(BinaryStatement statement)
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
			Instruction.Modulo => new Instance(right.ReturnType,
				Convert.ToDouble(left.Value) % Convert.ToDouble(right.Value)),
			_ => Registers[statement.Registers[^1]] //ncrunch: no coverage
		};
	}

	private static Instance GetAdditionResult(Instance left, Instance right)
	{
		var leftReturnTypeName = left.TypeName;
		var rightReturnTypeName = right.TypeName;
		if (leftReturnTypeName == Base.Number && rightReturnTypeName == Base.Number)
			return new Instance(right.ReturnType ?? left.ReturnType,
				Convert.ToDouble(left.Value) + Convert.ToDouble(right.Value));
		if (leftReturnTypeName == Base.Text && rightReturnTypeName == Base.Text)
			return new Instance(right.ReturnType ?? left.ReturnType,
				left.Value.ToString() + right.Value);
		if (rightReturnTypeName == Base.Text && leftReturnTypeName == Base.Number)
			return new Instance(right.ReturnType, left.Value.ToString() + right.Value);
		return new Instance(left.ReturnType, left.Value + right.Value.ToString());
	}

	private (Instance, Instance) GetOperands(BinaryStatement statement) =>
		Registers.Count < 2
			? throw new OperandsRequired()
			: (Registers[statement.Registers[1]], Registers[statement.Registers[0]]);

	private void TryConditionalOperationExecution(BinaryStatement statement)
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
			statement is JumpIfNotZeroStatement jumpIfNotZeroStatement &&
			Convert.ToInt32(Registers[jumpIfNotZeroStatement.Register].Value) > 0)
		{
			instructionIndex += Convert.ToInt32(((JumpIfStatement)statement).Steps);
		}
		else if (!conditionFlag && statement.Instruction is Instruction.JumpToIdIfFalse ||
			conditionFlag && statement.Instruction is Instruction.JumpToIdIfTrue)
		{
			var id = ((JumpToIdStatement)statement).Id;
			var endIndex = statements.IndexOf(statements.First(jumpStatement =>
				jumpStatement.Instruction is Instruction.JumpEnd &&
				jumpStatement is JumpToIdStatement jumpViaId && jumpViaId.Id == id));
			if (endIndex != -1)
				instructionIndex = endIndex;
		}
	}

	public class OperandsRequired : Exception { }
}