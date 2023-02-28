using System.Data;
using Strict.Language;
using Strict.Language.Expressions;

namespace Strict.VirtualMachine;

// ReSharper disable once ClassTooBig
public sealed class VirtualMachine
{
	public Memory Memory { get; set; } = new();
	private bool conditionFlag;
	private int instructionIndex;
	private IList<Statement> statements = new List<Statement>();
	public Instance? Returns { get; private set; }

	public VirtualMachine Execute(IList<Statement> allStatements)
	{
		Clear();
		return RunStatements(allStatements);
	}

	public VirtualMachine Invoke(IList<Statement> allStatements) => RunStatements(allStatements);

	private VirtualMachine RunStatements(IList<Statement> allStatements)
	{
		statements = allStatements;
		for (instructionIndex = 0; instructionIndex != -1 && instructionIndex < allStatements.Count;
			instructionIndex++)
			ExecuteStatement(allStatements[instructionIndex]);
		return this;
	}

	private void Clear()
	{
		conditionFlag = false;
		instructionIndex = 0;
		statements.Clear();
		Returns = null;
		Memory.Registers.Clear();
		Memory.Variables.Clear();
	}

	private void ExecuteStatement(Statement statement)
	{
		if (TryExecuteReturn(statement))
			return;
		TryStoreInstructions(statement);
		TryLoadInstructions(statement);
		TryLoopInitInstruction(statement);
		TryLoopEndInstruction(statement);
		TryInvokeInstruction(statement);
		TryExecute(statement);
	}

	private void TryLoopEndInstruction(Statement statement)
	{
		if (statement is IterationEndStatement loopEnd)
		{
			var iterator = Memory.Registers[loopEnd.Register];
			iterator.Value = (int)iterator.Value - 1;
			if ((int)iterator.Value == 0)
				iteratorInitialized = false;
		}
	}

	private void TryInvokeInstruction(Statement statement)
	{
		if (statement is not InvokeStatement { MethodCall: { } } invokeStatement)
			return;
		FormArgumentsForMethodCall(invokeStatement);
		var methodStatements = GetByteCodeFromInvokedMethodCall(invokeStatement);
		var instance = RunAndGetResultFromInvokedMethodCall(methodStatements);
		if (instance != null)
			Memory.Registers[invokeStatement.Register] = instance;
	}

	private Instance? RunAndGetResultFromInvokedMethodCall(IList<Statement> methodStatements)
	{
		var members =
			new Dictionary<string, Instance>(
				Memory.Variables.Where(variable => variable.Value.IsMember));
		var instance =
			new VirtualMachine
			{
				Memory = new Memory { Registers = Memory.Registers, Variables = members }
			}.Invoke(methodStatements).Returns;
		return instance;
	}

	private List<Statement> GetByteCodeFromInvokedMethodCall(
		InvokeStatement invokeStatement)
	{
		if (invokeStatement.PersistedRegistry == null || invokeStatement.MethodCall == null)
			throw new InvalidExpressionException(); //TODO: Cover this line ncrunch: no coverage
		return invokeStatement.MethodCall.Instance == null
			? new ByteCodeGenerator(
					new InvokedMethod(
						((Body)invokeStatement.MethodCall.Method.GetBodyAndParseIfNeeded()).Expressions,
						FormArgumentsForMethodCall(invokeStatement)), invokeStatement.PersistedRegistry).
				Generate()
			: new ByteCodeGenerator(
				new InstanceInvokedMethod(
					((Body)invokeStatement.MethodCall.Method.GetBodyAndParseIfNeeded()).Expressions,
					FormArgumentsForMethodCall(invokeStatement), invokeStatement.MethodCall.Instance),
				invokeStatement.PersistedRegistry).Generate();
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
				: Memory.Variables[argument.ToString()];
			arguments.Add(invokeStatement.MethodCall.Method.Parameters[index].Name, argumentInstance);
		}
		return arguments;
	}

	private bool TryExecuteReturn(Statement statement)
	{
		if (statement is not ReturnStatement returnStatement)
			return false;
		Returns = Memory.Registers[returnStatement.Register];
		if (!Returns.Value.GetType().IsPrimitive && Returns.Value is not Value)
			return false;
		instructionIndex = -2;
		return true;
	}

	private bool iteratorInitialized;

	private void TryLoopInitInstruction(Statement statement)
	{
		if (statement is not LoopBeginStatement initLoopStatement)
			return;
		ProcessLoopIndex();
		Memory.Variables.TryGetValue(initLoopStatement.Identifier, out var iterableVariable);
		if (iterableVariable == null)
			return;
		if (!iteratorInitialized)
			InitializeIterator(initLoopStatement,
				iterableVariable); //TODO: Get rid of this and figure out something better. (LM)
		AlterValueVariable(iterableVariable);
	}

	private void ProcessLoopIndex()
	{
		if (Memory.Variables.ContainsKey("index"))
			Memory.Variables["index"].Value = Convert.ToInt32(Memory.Variables["index"].Value) + 1;
		else
			Memory.Variables.Add("index", new Instance(Base.Number, 0));
	}

	private void InitializeIterator(LoopBeginStatement initLoopStatement, Instance iterableVariable)
	{
		Memory.Registers[initLoopStatement.Register] =
			new Instance(Base.Number, GetLength(iterableVariable));
		iteratorInitialized = true;
	}

	private static int GetLength(Instance iterableInstance)
	{
		if (iterableInstance.Value is string iterableString)
			return iterableString.Length;
		if (iterableInstance.Value is int or double)
			return Convert.ToInt32(iterableInstance.Value);
		if (iterableInstance.ReturnType != null && iterableInstance.ReturnType.IsIterator)
			return ((IEnumerable<Expression>)iterableInstance.Value).Count();
		return 0; //ncrunch: no coverage
	}

	private void AlterValueVariable(Instance iterableVariable)
	{
		var index = Convert.ToInt32(Memory.Variables["index"].Value);
		var value = iterableVariable.Value.ToString();
		if (iterableVariable.ReturnType?.Name == Base.Text && value != null)
			Memory.Variables["value"] = new Instance(Base.Number, value[index].ToString());
		else if (iterableVariable.ReturnType is GenericTypeImplementation genericIterable &&
			genericIterable.Generic.Name == Base.List)
			Memory.Variables["value"] = new Instance(((List<Expression>)iterableVariable.Value)[index]);
		else if (iterableVariable.ReturnType?.Name == Base.Number)
			Memory.Variables["value"] =
				new Instance(Base.Number, Convert.ToInt32(iterableVariable.Value) + index);
	}

	private void TryStoreInstructions(Statement statement)
	{
		if (statement.Instruction > Instruction.SetLoadSeparator)
			return;
		if (statement is SetStatement setStatement)
			Memory.Registers[setStatement.Register] = setStatement.Instance;
		else if (statement is StoreVariableStatement storeVariableStatement)
			Memory.Variables[storeVariableStatement.Identifier] = storeVariableStatement.Instance;
		else if (statement is StoreFromRegisterStatement storeFromRegisterStatement)
			Memory.Variables[storeFromRegisterStatement.Identifier] =
				Memory.Registers[storeFromRegisterStatement.Register];
	}

	private void TryLoadInstructions(Statement statement)
	{
		if (statement is LoadVariableStatement loadVariableStatement)
			LoadVariableIntoRegister(loadVariableStatement);
		else if (statement is LoadConstantStatement loadConstantStatement)
			Memory.Registers[loadConstantStatement.Register] = loadConstantStatement.Instance;
	}

	private void LoadVariableIntoRegister(LoadVariableStatement statement) =>
		Memory.Registers[statement.Register] = Memory.Variables[statement.Identifier];

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
		Memory.Registers[statement.Registers[^1]] = statement.Instruction switch
		{
			Instruction.Add => GetAdditionResult(left, right),
			Instruction.Subtract => GetSubtractionResult(left, right),
			Instruction.Multiply => new Instance(right.ReturnType,
				Convert.ToDouble(left.Value) * Convert.ToDouble(right.Value)),
			Instruction.Divide => new Instance(right.ReturnType,
				Convert.ToDouble(left.Value) / Convert.ToDouble(right.Value)),
			Instruction.Modulo => new Instance(right.ReturnType,
				Convert.ToDouble(left.Value) % Convert.ToDouble(right.Value)),
			_ => Memory.Registers[statement.Registers[^1]] //ncrunch: no coverage
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
		return !leftReturnTypeName.EndsWith('s')
			? new Instance(left.ReturnType, left.Value + right.Value.ToString())
			: AddElementToTheListAndGetInstance(left, right);
	}

	private static Instance AddElementToTheListAndGetInstance(Instance left, Instance right)
	{
		var elements = new List<Expression>((List<Expression>)left.Value);
		if (right.Value is Expression rightExpression)
			elements.Add(rightExpression);
		else
		{
			var rightValue = new Value(elements.First().ReturnType, right.Value);
			elements.Add(rightValue);
		}
		return new Instance(left.ReturnType, elements);
	}

	private static Instance GetSubtractionResult(Instance left, Instance right)
	{
		if (!left.TypeName.EndsWith('s'))
			return new Instance(left.ReturnType,
				Convert.ToDouble(left.Value) - Convert.ToDouble(right.Value));
		var elements = new List<Expression>((List<Expression>)left.Value);
		if (right.Value is Expression rightExpression)
			elements.Remove(rightExpression);
		else
		{
			var indexToRemove =
				elements.FindIndex(element => ((Value)element).Data.Equals(right.Value));
			elements.RemoveAt(indexToRemove);
		}
		return new Instance(left.ReturnType, elements);
	}

	private (Instance, Instance) GetOperands(BinaryStatement statement) =>
		Memory.Registers.Count < 2
			? throw new OperandsRequired()
			: (Memory.Registers[statement.Registers[1]], Memory.Registers[statement.Registers[0]]);

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
			Convert.ToInt32(Memory.Registers[jumpIfNotZeroStatement.Register].Value) > 0)
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