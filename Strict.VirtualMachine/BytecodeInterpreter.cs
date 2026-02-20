using System.Diagnostics;
using Strict.Expressions;
using Strict.Language;
using Strict.Runtime.Statements;
using BinaryStatement = Strict.Runtime.Statements.Binary;
using Return = Strict.Runtime.Statements.Return;

namespace Strict.Runtime;

public sealed class BytecodeInterpreter
{
	public BytecodeInterpreter Execute(IList<Statement> allStatements)
	{
		Clear();
		return RunStatements(allStatements);
	}

	private void Clear()
	{
		conditionFlag = false;
		instructionIndex = 0;
		iteratorInitialized = false;
		loopIterationNumber = 0;
		statements.Clear();
		Returns = null;
		Memory.Registers.Clear();
		Memory.Variables.Clear();
	}

	private bool conditionFlag;
	private int instructionIndex;
	private bool iteratorInitialized;
	private int loopIterationNumber;
	private IList<Statement> statements = new List<Statement>();
	public Instance? Returns { get; private set; }
	// ReSharper disable once AutoPropertyCanBeMadeGetOnly.Local
	public Memory Memory { get; private init; } = new();

	private BytecodeInterpreter RunStatements(IList<Statement> allStatements)
	{
		statements = allStatements;
		for (instructionIndex = 0;
			instructionIndex is not -1 && instructionIndex < allStatements.Count; instructionIndex++)
			ExecuteStatement(allStatements[instructionIndex]);
		return this;
	}

	private void ExecuteStatement(Statement statement)
	{
		Debug.Assert(statement != null);
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
			list.RemoveAll(expression => EqualsExtensions.AreEqual(((Value)expression).Data, item));
		}
		else if (statement is RemoveFromTableStatement removeFromTableStatement)
		{
			var key = Memory.Registers[removeFromTableStatement.Register].GetRawValue();
			var dict = (Dictionary<Value, Value>)Memory.Variables[removeFromTableStatement.Identifier].Value;
			var keyToRemove = dict.Keys.FirstOrDefault(k => EqualsExtensions.AreEqual(k.Data, key));
			if (keyToRemove != null)
				dict.Remove(keyToRemove);
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
		if (statement is not Conversion conversionStatement)
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
		if (statement is not IterationEnd iterationEndStatement)
			return;
		loopIterationNumber--;
		if (loopIterationNumber <= 0)
			return;
		instructionIndex -= iterationEndStatement.Steps + 1;
	}

	private void TryInvokeInstruction(Statement statement)
	{
		if (statement is not Invoke { Method: not null } invokeStatement)
			return;
		if (TryCreateEmptyDictionaryInstance(invokeStatement))
			return;
		if (GetValueByKeyForDictionaryAndStoreInRegister(invokeStatement))
			return;
		var methodStatements = GetByteCodeFromInvokedMethodCall(invokeStatement);
		var instance = new BytecodeInterpreter
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

	private bool TryCreateEmptyDictionaryInstance(Invoke invoke)
	{
		if (invoke.Method?.Instance != null ||
			invoke.Method?.Method.Name != Method.From ||
			invoke.Method?.ReturnType is not GenericTypeImplementation
			{
				Generic.Name: Base.Dictionary
			} dictionaryType)
			return false;
		Memory.Registers[invoke.Register] = new Instance(dictionaryType, new Dictionary<Value, Value>());
		return true;
	}

	private bool GetValueByKeyForDictionaryAndStoreInRegister(Invoke invoke)
	{
		if (invoke.Method?.Method.Name != "Get" ||
			invoke.Method.Instance?.ReturnType is not GenericTypeImplementation
			{
				Generic.Name: Base.Dictionary
			})
			return false;
		var keyArg = invoke.Method.Arguments[0];
		var keyData = keyArg is Value argValue
			? argValue.Data
			: Memory.Variables[keyArg.ToString()].Value;
		var dictionary = Memory.Variables[invoke.Method.Instance.ToString()].Value;
		var value = ((Dictionary<Value, Value>)dictionary).
			FirstOrDefault(element => EqualsExtensions.AreEqual(element.Key.Data, keyData)).Value;
		if (value != null)
			Memory.Registers[invoke.Register] = new Instance(value.ReturnType, value);
		return true;
	}

	private List<Statement> GetByteCodeFromInvokedMethodCall(Invoke invoke)
	{
		if (invoke.Method?.Instance == null &&
			invoke.Method?.Method != null && invoke.PersistedRegistry != null)
			return new ByteCodeGenerator(
					new InvokedMethod(
						GetExpressionsFromMethod(invoke.Method.Method),
						FormArgumentsForMethodCall(invoke)), invoke.PersistedRegistry).
				Generate();
		var instance = GetVariableInstanceFromMemory(invoke.Method?.Instance?.ToString() ?? throw new InvalidOperationException());
		return new ByteCodeGenerator(
			new InstanceInvokedMethod(
				GetExpressionsFromMethod(invoke.Method!.Method),
				FormArgumentsForMethodCall(invoke), instance),
			invoke.PersistedRegistry ?? throw new InvalidOperationException()).Generate();
	}

	private static IReadOnlyList<Expression> GetExpressionsFromMethod(Method method)
	{
		var result = method.GetBodyAndParseIfNeeded();
		return result is Body body ? body.Expressions : [result];
	}

	private Instance GetVariableInstanceFromMemory(string variableIdentifier)
	{
		Memory.Variables.TryGetValue(variableIdentifier, out var methodCallInstance);
		if (methodCallInstance == null)
			throw new VariableNotFoundInMemory();
		return methodCallInstance;
	}

	private Dictionary<string, Instance> FormArgumentsForMethodCall(Invoke invoke)
	{
		var arguments = new Dictionary<string, Instance>();
		if (invoke.Method == null)
			return arguments; // ncrunch: no coverage
		for (var index = 0; index < invoke.Method.Method.Parameters.Count; index++)
		{
			var argument = invoke.Method.Arguments[index];
			var argumentInstance = argument is Value argumentValue
				? new Instance(argumentValue.ReturnType, argumentValue.Data)
				: Memory.Variables[argument.ToString()];
			arguments.Add(invoke.Method.Method.Parameters[index].Name, argumentInstance);
		}
		return arguments;
	}

	private bool TryExecuteReturn(Statement statement)
	{
		if (statement is not Return returnStatement)
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
		ProcessLoopIndex();
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
		if (statement is not LoopRangeBeginStatement loopRangeStatement)
			return;
		if (Memory.Variables.ContainsKey("index"))
			Memory.Variables["index"].Value = Convert.ToInt32(Memory.Variables["index"].Value) +
				(IsRangeDecreasing(loopRangeStatement)
					? -1
					: 1);
		else
			Memory.Variables.Add("index", Memory.Registers[loopRangeStatement.StartIndex]);
		Memory.Variables["value"] = Memory.Variables["index"];
		if (!iteratorInitialized)
			InitializeRangeIterator(loopRangeStatement);
	}

	private bool IsRangeDecreasing(LoopRangeBeginStatement loopRangeStatement) =>
		Memory.Registers[loopRangeStatement.EndIndex] <
		Memory.Registers[loopRangeStatement.StartIndex];

	private void ProcessLoopIndex()
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

	private void InitializeRangeIterator(LoopRangeBeginStatement loopRangeBeginStatement)
	{
		var startIndex = Convert.ToInt32(Memory.Registers[loopRangeBeginStatement.StartIndex].Value);
		var endIndex = Convert.ToInt32(Memory.Registers[loopRangeBeginStatement.EndIndex].Value);
		loopIterationNumber = (IsRangeDecreasing(loopRangeBeginStatement)
			? startIndex - endIndex
			: endIndex - startIndex) + 1;
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
		if (statement.Instruction > Instruction.StoreSeparator)
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
		if (statement is LoadVariableToRegister loadVariableStatement)
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
		else if (statement is Jump jumpStatement)
			TryJumpOperation(jumpStatement);
		else if (statement is JumpIf jumpIfStatement)
			TryJumpIfOperation(jumpIfStatement);
		else if (statement is JumpToId jumpToIdStatement)
			TryJumpToIdOperation(jumpToIdStatement);
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
			Instruction.Equal => EqualsExtensions.AreEqual(left.GetRawValue(), right.GetRawValue()),
			Instruction.NotEqual => !EqualsExtensions.AreEqual(left.GetRawValue(), right.GetRawValue()),
			_ => false //ncrunch: no coverage
		};
	}

	private static void NormalizeValues(params Instance[] instances)
	{
		foreach (var instance in instances)
			if (instance.Value is MemberCall member && member.Member.InitialValue != null)
				instance.Value = member.Member.InitialValue;
	}

	private void TryJumpOperation(Jump statement)
	{
		if (conditionFlag && statement.Instruction is Instruction.JumpIfTrue ||
			!conditionFlag && statement.Instruction is Instruction.JumpIfFalse)
			instructionIndex += statement.InstructionsToSkip;
	}

	private void TryJumpIfOperation(JumpIf statement)
	{
		if (conditionFlag && statement.Instruction is Instruction.JumpIfTrue ||
			!conditionFlag && statement.Instruction is Instruction.JumpIfFalse ||
			statement is JumpIfNotZero jumpIfNotZeroStatement &&
			Convert.ToInt32(Memory.Registers[jumpIfNotZeroStatement.Register].Value) > 0)
			instructionIndex += Convert.ToInt32(statement.Steps);
	}

	private void TryJumpToIdOperation(JumpToId statement)
	{
		if (!conditionFlag && statement.Instruction is Instruction.JumpToIdIfFalse ||
			conditionFlag && statement.Instruction is Instruction.JumpToIdIfTrue)
		{
			var id = statement.Id;
			var endIndex = statements.IndexOf(statements.First(jumpStatement =>
				jumpStatement.Instruction is Instruction.JumpEnd &&
				jumpStatement is JumpToId jumpViaId && jumpViaId.Id == id));
			if (endIndex != -1)
				instructionIndex = endIndex;
		}
	}

	public sealed class OperandsRequired : Exception { }
	private sealed class VariableNotFoundInMemory : Exception { }
}