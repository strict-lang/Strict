using Strict.Language;
using Strict.Language.Expressions;

namespace Strict.VirtualMachine;

public sealed class VirtualMachine
{
	private bool conditionFlag;
	private int instructionIndex;
	private bool iteratorInitialized;
	private int loopIterationNumber;
	private IList<Statement> statements = new List<Statement>();
	public Memory Memory { get; private init; } = new();
	public Instance? Returns { get; private set; }

	public VirtualMachine Execute(IList<Statement> allStatements)
	{
		Clear();
		return RunStatements(allStatements);
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

	private VirtualMachine RunStatements(IList<Statement> allStatements)
	{
		statements = allStatements;
		for (instructionIndex = 0;
			instructionIndex is not -1 && instructionIndex < allStatements.Count; instructionIndex++)
			ExecuteStatement(allStatements[instructionIndex]);
		return this;
	}

	private void ExecuteStatement(Statement statement)
	{
		if (TryExecuteReturn(statement))
			return;
		TryStoreInstructions(statement);
		TryLoadInstructions(statement);
		TryLoopInitInstruction(statement);
		TryLoopRangeInitInstruction(statement);
		TryLoopEndInstruction(statement);
		TryInvokeInstruction(statement);
		TryWriteToListInstruction(statement);
		TryConversionStatement(statement);
		TryWriteToTableInstruction(statement);
		TryRemoveStatement(statement);
		TryExecuteListCall(statement);
		TryExecuteRest(statement);
	}

	private void TryRemoveStatement(Statement statement)
	{
		if (statement is RemoveStatement removeStatement)
		{
			var item = Memory.Registers[removeStatement.Register].GetRawValue();
			var list = (List<Expression>)Memory.Variables[removeStatement.Identifier].Value;
			list.RemoveAll(expression => ((Value)expression).Data.Equals(item));
		}
		else if (statement is RemoveFromTableStatement removeFromTableStatement)
		{
			Memory.Registers[removeFromTableStatement.Register].GetRawValue();
		}
	}

	private void TryExecuteListCall(Statement statement)
	{
		if (statement is not ListCallStatement listCallStatement)
			return;
		var indexValue =
			Convert.ToInt32(Memory.Registers[listCallStatement.IndexValueRegister].GetRawValue());
		var variableListElement =
			((List<Expression>)Memory.Variables[listCallStatement.Identifier].Value).
			ElementAt(indexValue);
		Memory.Registers[listCallStatement.Register] =
			new Instance(variableListElement.ReturnType, variableListElement);
	}

	private void TryConversionStatement(Statement statement)
	{
		if (statement is not ConversionStatement conversionStatement)
			return;
		var instanceToBeConverted = Memory.Registers[conversionStatement.Register].GetRawValue();
		if (conversionStatement.ConversionType.Name == Base.Text)
			Memory.Registers[conversionStatement.RegisterToStoreConversion] = new Instance(
				conversionStatement.ConversionType,
				instanceToBeConverted.ToString() ?? throw new InvalidOperationException());
		else if (conversionStatement.ConversionType.Name == Base.Number)
			Memory.Registers[conversionStatement.RegisterToStoreConversion] = new Instance(
				conversionStatement.ConversionType, Convert.ToDecimal(instanceToBeConverted));
	}

	private void TryWriteToTableInstruction(Statement statement)
	{
		if (statement is not WriteToTableStatement writeToTableStatement)
			return;
		Memory.AddToDictionary(writeToTableStatement.Identifier,
			Memory.Registers[writeToTableStatement.Key], Memory.Registers[writeToTableStatement.Value]);
	}

	private void TryWriteToListInstruction(Statement statement)
	{
		if (statement is not WriteToListStatement writeToListStatement)
			return;
		Memory.AddToCollectionVariable(writeToListStatement.Identifier,
			Memory.Registers[writeToListStatement.Register].Value);
	}

	private void TryLoopEndInstruction(Statement statement)
	{
		if (statement is not IterationEndStatement iterationEndStatement)
			return;
		loopIterationNumber--;
		if (loopIterationNumber <= 0)
			return;
		instructionIndex -= iterationEndStatement.Steps + 1;
	}

	private void TryInvokeInstruction(Statement statement)
	{
		if (statement is not InvokeStatement { MethodCall: not null } invokeStatement)
			return;
		if (GetValueByKeyForDictionaryAndStoreInRegister(invokeStatement))
			return;
		var methodStatements = GetByteCodeFromInvokedMethodCall(invokeStatement);
		var instance = new VirtualMachine
		{
			Memory = new Memory
			{
				Registers = Memory.Registers,
				Variables =
					new Dictionary<string, Instance>(
						Memory.Variables.Where(variable => variable.Value.IsMember))
			}
		}.RunStatements(methodStatements).Returns;
		if (instance != null)
			Memory.Registers[invokeStatement.Register] = instance;
	}

	private bool GetValueByKeyForDictionaryAndStoreInRegister(InvokeStatement invokeStatement)
	{
		if (invokeStatement.MethodCall?.Method.Name != "Get" ||
			invokeStatement.MethodCall.Instance?.ReturnType is not GenericTypeImplementation
			{
				Generic.Name: Base.Dictionary
			})
			return false;
		var key = (Value)invokeStatement.MethodCall.Arguments[0];
		var dictionary = Memory.Variables[invokeStatement.MethodCall.Instance.ToString()].Value;
		var value = ((Dictionary<Value, Value>)dictionary).
			FirstOrDefault(element => element.Key.Data.Equals(key.Data)).Value;
		if (value != null)
			Memory.Registers[invokeStatement.Register] = new Instance(value.ReturnType, value);
		return true;
	}

	private List<Statement> GetByteCodeFromInvokedMethodCall(InvokeStatement invokeStatement)
	{
		if (invokeStatement.MethodCall?.Instance == null &&
			invokeStatement.MethodCall?.Method != null && invokeStatement.PersistedRegistry != null)
			return new ByteCodeGenerator(
					new InvokedMethod(
						((Body)invokeStatement.MethodCall.Method.GetBodyAndParseIfNeeded()).Expressions,
						FormArgumentsForMethodCall(invokeStatement)), invokeStatement.PersistedRegistry).
				Generate();
		var instance = GetVariableInstanceFromMemory(invokeStatement.MethodCall?.Instance?.ToString() ?? throw new InvalidOperationException());
		return new ByteCodeGenerator(
			new InstanceInvokedMethod(
				((Body)invokeStatement.MethodCall.Method.GetBodyAndParseIfNeeded()).Expressions,
				FormArgumentsForMethodCall(invokeStatement), instance),
			invokeStatement.PersistedRegistry ?? throw new InvalidOperationException()).Generate();
	}

	private Instance GetVariableInstanceFromMemory(string variableIdentifier)
	{
		Memory.Variables.TryGetValue(variableIdentifier, out var methodCallInstance);
		if (methodCallInstance == null)
			throw new VariableNotFoundInMemory();
		return methodCallInstance;
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

	private void TryLoopInitInstruction(Statement statement)
	{
		if (statement is not LoopBeginStatement initLoopStatement)
			return;
		ProcessLoopIndex(initLoopStatement);
		Memory.Registers.TryGetValue(initLoopStatement.Register, out var iterableVariable);
		if (iterableVariable is null)
			return; //ncrunch: no coverage
		if (!iteratorInitialized)
			InitializeIterator(
				iterableVariable);
		AlterValueVariable(iterableVariable);
	}

	private void TryLoopRangeInitInstruction(Statement statement)
	{
		if (statement is not LoopBeginStatementRange initLoopStatement)
			return;
		if (Memory.Variables.ContainsKey("index"))
			Memory.Variables["index"].Value = Convert.ToInt32(Memory.Variables["index"].Value) + 1;
		else
			Memory.Variables.Add("index", Memory.Registers[initLoopStatement.StartIndex]);
		Memory.Variables["value"] = Memory.Variables["index"];
		if (!iteratorInitialized)
			InitializeIterator(Memory.Registers[initLoopStatement.EndIndex] -
				Memory.Registers[initLoopStatement.StartIndex]);

	}

	private void ProcessLoopIndex(LoopBeginStatement statement)
	{
		if (Memory.Variables.ContainsKey("index"))
			Memory.Variables["index"].Value = Convert.ToInt32(Memory.Variables["index"].Value) + 1;
		else
			Memory.Variables.Add("index", new Instance(Base.Number, 0));
	}

	private void InitializeIterator(Instance iterableVariable)
	{
		loopIterationNumber = GetLength(iterableVariable);
		iteratorInitialized = true;
	}


	private static int GetLength(Instance iterableInstance)
	{
		if (iterableInstance.Value is string iterableString)
			return iterableString.Length;
		if (iterableInstance.Value is int or double)
			return Convert.ToInt32(iterableInstance.Value);
		return iterableInstance.ReturnType is { IsIterator: true }
			? ((IEnumerable<Expression>)iterableInstance.Value).Count()
			: 0; //ncrunch: no coverage
	}

	private void AlterValueVariable(Instance iterableVariable)
	{
		var index = Convert.ToInt32(Memory.Variables["index"].Value);
		var value = iterableVariable.Value.ToString();
		if (iterableVariable.ReturnType?.Name == Base.Text && value is not null)
			Memory.Variables["value"] = new Instance(Base.Text, value[index].ToString());
		else if (iterableVariable.ReturnType is GenericTypeImplementation { Generic.Name: Base.List })
			Memory.Variables["value"] = new Instance(((List<Expression>)iterableVariable.Value)[index]);
		else if (iterableVariable.ReturnType?.Name == Base.Number)
			Memory.Variables["value"] =
				new Instance(Base.Number, index + 1);
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
			Memory.Registers[loadVariableStatement.Register] =
				Memory.Variables[loadVariableStatement.Identifier];
		else if (statement is LoadConstantStatement loadConstantStatement)
			Memory.Registers[loadConstantStatement.Register] = loadConstantStatement.Instance;
	}

	private void TryExecuteRest(Statement statement)
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
			Instruction.Add => left + right,
			Instruction.Subtract => left - right,
			Instruction.Multiply => new Instance(right.ReturnType,
				Convert.ToDouble(left.Value) * Convert.ToDouble(right.Value)),
			Instruction.Divide => new Instance(right.ReturnType,
				Convert.ToDouble(left.Value) / Convert.ToDouble(right.Value)),
			Instruction.Modulo => new Instance(right.ReturnType,
				Convert.ToDouble(left.Value) % Convert.ToDouble(right.Value)),
			_ => Memory.Registers[statement.Registers[^1]] //ncrunch: no coverage
		};
	}

	private (Instance, Instance) GetOperands(BinaryStatement statement) =>
		Memory.Registers.Count < 2
			? throw new OperandsRequired()
			: (Memory.Registers[statement.Registers[1]], Memory.Registers[statement.Registers[0]]);

	private void TryConditionalOperationExecution(BinaryStatement statement)
	{
		var (right, left) = GetOperands(statement);
		NormalizeValues(right, left);
		conditionFlag = statement.Instruction switch
		{
			Instruction.GreaterThan => left > right,
			Instruction.LessThan => left < right,
			Instruction.Equal => left.Value.Equals(right.Value),
			Instruction.NotEqual => !left.Value.Equals(right.Value),
			_ => false //ncrunch: no coverage
		};
	}

	private static void NormalizeValues(params Instance[] instances)
	{
		foreach (var instance in instances)
		{
			if (instance.Value is not MemberCall member)
				continue;
			if (member.Member.Value != null)
				instance.Value = member.Member.Value;
		}
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

	public sealed class OperandsRequired : Exception { }
	private sealed class VariableNotFoundInMemory : Exception { }
}