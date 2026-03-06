using System.Diagnostics;
using Strict.Expressions;
using Strict.Language;
using Strict.Runtime.Statements;
using BinaryStatement = Strict.Runtime.Statements.Binary;
using Return = Strict.Runtime.Statements.Return;
using Type = Strict.Language.Type;

namespace Strict.Runtime;

public sealed class BytecodeInterpreter
{
	public BytecodeInterpreter Execute(IList<Statement> allStatements)
	{
		Clear();
		foreach (var loopBegin in allStatements.OfType<LoopBeginStatement>())
			loopBegin.Reset();
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

	private bool conditionFlag;
	private int instructionIndex;
	private IList<Statement> statements = new List<Statement>();
	public ValueInstance? Returns { get; private set; }
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
		TryLoopEndInstruction(statement);
		TryInvokeInstruction(statement);
		TryWriteToListInstruction(statement);
		TryWriteToTableInstruction(statement);
		TryRemoveStatement(statement);
		TryExecuteListCall(statement);
		TryExecuteRest(statement);
	}

	private void TryRemoveStatement(Statement statement)
	{
		if (statement is not RemoveStatement removeStatement)
			return;
		var item = Memory.Registers[removeStatement.Register];
		var listItems = new List<ValueInstance>(Memory.Variables[removeStatement.Identifier].List.Items);
		listItems.RemoveAll(vi => AreEqual(vi, item));
		Memory.Variables[removeStatement.Identifier] = new ValueInstance(
			Memory.Variables[removeStatement.Identifier].List.ReturnType, listItems);
	}

	private void TryExecuteListCall(Statement statement)
	{
		if (statement is not ListCallStatement listCallStatement)
			return;
		var indexValue = (int)Memory.Registers[listCallStatement.IndexValueRegister].Number;
		var variableListElement = Memory.Variables[listCallStatement.Identifier].List.Items.ElementAt(indexValue);
		Memory.Registers[listCallStatement.Register] = variableListElement;
	}

	private void TryWriteToListInstruction(Statement statement)
	{
		if (statement is not WriteToListStatement writeToListStatement)
			return;
		Memory.AddToCollectionVariable(writeToListStatement.Identifier,
			Memory.Registers[writeToListStatement.Register]);
	}

	private void TryWriteToTableInstruction(Statement statement)
	{
		if (statement is not WriteToTableStatement writeToTableStatement)
			return;
		Memory.AddToDictionary(writeToTableStatement.Identifier,
			Memory.Registers[writeToTableStatement.Key], Memory.Registers[writeToTableStatement.Value]);
	}

	private void TryLoopEndInstruction(Statement statement)
	{
		if (statement is not LoopEndStatement loopEndStatement)
			return;
		var loopBegin = statements.Take(instructionIndex).OfType<LoopBeginStatement>().Last();
		loopBegin.LoopCount--;
		if (loopBegin.LoopCount <= 0)
			return;
		instructionIndex -= loopEndStatement.Steps + 1;
	}

	private void TryInvokeInstruction(Statement statement)
	{
		if (statement is not Invoke { Method: not null } invokeStatement)
			return;
		if (TryCreateEmptyDictionaryInstance(invokeStatement))
			return;
		if (TryHandleToConversion(invokeStatement))
			return;
		if (TryHandleIncrementDecrement(invokeStatement))
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
					new Dictionary<string, ValueInstance>(
						Memory.Variables.Where(v => Memory.MemberVariableNames.Contains(v.Key)))
			}
		}.RunStatements(methodStatements).Returns;
		if (instance != null)
			Memory.Registers[invokeStatement.Register] = instance.Value;
	}

	private bool TryHandleIncrementDecrement(Invoke invoke)
	{
		var methodName = invoke.Method?.Method.Name;
		if (methodName != "Increment" && methodName != "Decrement")
			return false;
		if (invoke.Method!.Instance == null ||
			!Memory.Variables.TryGetValue(invoke.Method.Instance.ToString(), out var current))
			return false; //ncrunch: no coverage
		var delta = methodName == "Increment"
			? 1.0
			: -1.0;
		Memory.Registers[invoke.Register] =
			new ValueInstance(current.GetTypeExceptText(), current.Number + delta);
		return true;
	}

	private bool TryHandleToConversion(Invoke invoke)
	{
		if (invoke.Method?.Method.Name != BinaryOperator.To)
			return false;
		var instanceExpr = invoke.Method.Instance ?? throw new InvalidOperationException();
		var rawValue = instanceExpr is Value constValue
			? constValue.Data
			: Memory.Variables.TryGetValue(instanceExpr.ToString(), out var varValue)
				? varValue
				: throw new InvalidOperationException(); //ncrunch: no coverage
		var conversionType = invoke.Method.ReturnType;
		if (conversionType.IsText)
			Memory.Registers[invoke.Register] =
				rawValue.IsText
					? rawValue
					: new ValueInstance(rawValue.ToExpressionCodeString());
		else if (conversionType.IsNumber)
			Memory.Registers[invoke.Register] =
				rawValue.IsText
					? new ValueInstance(conversionType, Convert.ToDouble(rawValue.Text))
					: rawValue;
		return true;
	}

	private bool TryCreateEmptyDictionaryInstance(Invoke invoke)
	{
		if (invoke.Method?.Instance != null || invoke.Method?.Method.Name != Method.From ||
			invoke.Method?.ReturnType is not GenericTypeImplementation
			{
				Generic.Name: Type.Dictionary
			} dictionaryType)
			return false;
		Memory.Registers[invoke.Register] = new ValueInstance(dictionaryType, new Dictionary<ValueInstance, ValueInstance>());
		return true;
	}

	private bool GetValueByKeyForDictionaryAndStoreInRegister(Invoke invoke)
	{
		if (invoke.Method?.Method.Name != "Get" ||
			invoke.Method.Instance?.ReturnType is not GenericTypeImplementation
			{
				Generic.Name: Type.Dictionary
			})
			return false;
		var keyArg = invoke.Method.Arguments[0];
		var keyData = keyArg is Value argValue
			? argValue.Data
			: Memory.Variables[keyArg.ToString()];
		var dictionary = Memory.Variables[invoke.Method.Instance.ToString()];
		var value = dictionary.GetDictionaryItems().
			FirstOrDefault(element => element.Key.Equals(keyData)).Value;
		if (!Equals(value, default(ValueInstance)))
			Memory.Registers[invoke.Register] = value;
		return true;
	}

	private List<Statement> GetByteCodeFromInvokedMethodCall(Invoke invoke)
	{
		if (invoke.Method?.Instance == null && invoke.Method?.Method != null &&
			invoke.PersistedRegistry != null)
			return new ByteCodeGenerator(
				new InvokedMethod(GetExpressionsFromMethod(invoke.Method.Method),
					FormArgumentsForMethodCall(invoke), invoke.Method.Method.ReturnType),
				invoke.PersistedRegistry).Generate();
		if (!Memory.Variables.TryGetValue(invoke.Method?.Instance?.ToString() ??
			throw new InvalidOperationException(), out var instance))
			throw new VariableNotFoundInMemory(); //TODO: need test
		return new ByteCodeGenerator(
			new InstanceInvokedMethod(GetExpressionsFromMethod(invoke.Method!.Method),
				FormArgumentsForMethodCall(invoke), instance, invoke.Method.Method.ReturnType),
			invoke.PersistedRegistry ?? throw new InvalidOperationException()).Generate();
	}

	private static IReadOnlyList<Expression> GetExpressionsFromMethod(Method method)
	{
		var result = method.GetBodyAndParseIfNeeded();
		return result is Body body
			? body.Expressions
			: [result];
	}

	private Dictionary<string, ValueInstance> FormArgumentsForMethodCall(Invoke invoke)
	{
		var arguments = new Dictionary<string, ValueInstance>();
		if (invoke.Method == null)
			return arguments; // ncrunch: no coverage
		for (var index = 0; index < invoke.Method.Method.Parameters.Count; index++)
		{
			var argument = invoke.Method.Arguments[index];
			var argumentInstance = argument is Value argumentValue
				? argumentValue.Data
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
		instructionIndex = -2;
		return true;
	}

	private void TryLoopInitInstruction(Statement statement)
	{
		if (statement is not LoopBeginStatement loopBeginStatement)
			return;
		if (loopBeginStatement.IsRange)
			ProcessRangeLoopIteration(loopBeginStatement);
		else
			ProcessCollectionLoopIteration(loopBeginStatement);
	}

	private void ProcessCollectionLoopIteration(LoopBeginStatement loopBeginStatement)
	{
		if (!Memory.Registers.TryGetValue(loopBeginStatement.Register, out var iterableVariable))
			return; //ncrunch: no coverage
		Type numberType;
		if (iterableVariable.IsText)
		{
			var firstNumberVar = Memory.Variables.Values.FirstOrDefault(v => !v.IsText && !v.IsList && !v.IsDictionary);
			numberType = !Equals(firstNumberVar, default(ValueInstance))
				? firstNumberVar.GetTypeExceptText().GetType(Type.Number)
				: Memory.Registers.Values.
					FirstOrDefault(v => v is { IsText: false, IsList: false, IsDictionary: false }).
					GetTypeExceptText().GetType(Type.Number);
		}
		else
		{
			numberType = iterableVariable.GetTypeExceptText().GetType(Type.Number);
		}
		if (Memory.Variables.ContainsKey("index"))
		{
			var current = Memory.Variables["index"];
			Memory.Variables["index"] = new ValueInstance(numberType, current.Number + 1);
		}
		else
			Memory.Variables.Add("index", new ValueInstance(numberType, 0));
		if (!loopBeginStatement.IsInitialized)
		{
			loopBeginStatement.LoopCount = GetLength(iterableVariable);
			loopBeginStatement.IsInitialized = true;
		}
		AlterValueVariable(iterableVariable, numberType, loopBeginStatement);
		if (loopBeginStatement.LoopCount <= 0)
		{
			var stepsToLoopEnd = statements.Skip(instructionIndex + 1).
				TakeWhile(s => s is not LoopEndStatement).Count();
			instructionIndex += stepsToLoopEnd;
		}
	}

	private void ProcessRangeLoopIteration(LoopBeginStatement loopBeginStatement)
	{
		if (!loopBeginStatement.IsInitialized)
		{
			var startIndex = Convert.ToInt32(Memory.Registers[loopBeginStatement.Register].Number);
			var endIndex = Convert.ToInt32(Memory.Registers[loopBeginStatement.EndIndex!.Value].Number);
			loopBeginStatement.InitializeRangeState(startIndex, endIndex);
		}
		var isDecreasing = loopBeginStatement.IsDecreasing ?? false;
		var numberType = Memory.Registers[loopBeginStatement.Register].GetTypeExceptText().GetType(Type.Number);
		if (Memory.Variables.ContainsKey("index"))
		{
			var current = Memory.Variables["index"];
			Memory.Variables["index"] = new ValueInstance(numberType, current.Number + (isDecreasing
				? -1
				: 1));
		}
		else
			Memory.Variables.Add("index",
				new ValueInstance(numberType, loopBeginStatement.StartIndexValue ?? 0));
		Memory.Variables["value"] = Memory.Variables["index"];
	}

	private static int GetLength(ValueInstance iterableInstance)
	{
		if (iterableInstance.IsText)
			return iterableInstance.Text.Length;
		if (iterableInstance.IsList)
			return iterableInstance.List.Items.Count;
		return (int)iterableInstance.Number;
	}

	private void AlterValueVariable(ValueInstance iterableVariable, Type numberType,
		LoopBeginStatement loopBeginStatement)
	{
		var index = Convert.ToInt32(Memory.Variables["index"].Number);
		if (iterableVariable.IsText)
		{
			if (index < iterableVariable.Text.Length)
				Memory.Variables["value"] = new ValueInstance(iterableVariable.Text[index].ToString());
			else
				loopBeginStatement.LoopCount = 0;
			return;
		}
		if (iterableVariable.IsList)
		{
			var items = iterableVariable.List.Items;
			if (index < items.Count)
				Memory.Variables["value"] = items[index];
			else
				loopBeginStatement.LoopCount = 0;
			return;
		}
		Memory.Variables["value"] = new ValueInstance(numberType, index + 1);
	}

	private void TryStoreInstructions(Statement statement)
	{
		if (statement.Instruction > Instruction.StoreSeparator)
			return;
		if (statement is SetStatement setStatement)
			Memory.Registers[setStatement.Register] = setStatement.ValueInstance;
		else if (statement is StoreVariableStatement storeVariableStatement)
		{
			Memory.Variables[storeVariableStatement.Identifier] = storeVariableStatement.ValueInstance;
			if (storeVariableStatement.IsMember)
				Memory.MemberVariableNames.Add(storeVariableStatement.Identifier);
		}
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
			Memory.Registers[loadConstantStatement.Register] = loadConstantStatement.ValueInstance;
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
			Instruction.Add => AddValueInstances(left, right),
			Instruction.Subtract => SubtractValueInstances(left, right),
			Instruction.Multiply => new ValueInstance(right.GetTypeExceptText(),
				left.Number * right.Number),
			Instruction.Divide => new ValueInstance(right.GetTypeExceptText(),
				left.Number / right.Number),
			Instruction.Modulo => new ValueInstance(right.GetTypeExceptText(),
				left.Number % right.Number),
			_ => Memory.Registers[statement.Registers[^1]] //ncrunch: no coverage
		};
	}

	private static ValueInstance AddValueInstances(ValueInstance left, ValueInstance right)
	{
		if (left.IsList)
		{
			var items = new List<ValueInstance>(left.List.Items) { right };
			return new ValueInstance(left.List.ReturnType, items);
		}
		if (left.IsText || right.IsText)
			return new ValueInstance((left.IsText
				? left.Text
				: left.Number.ToString()) + (right.IsText
				? right.Text
				: right.Number.ToString()));
		return new ValueInstance(right.GetTypeExceptText(), left.Number + right.Number);
	}

	private static ValueInstance SubtractValueInstances(ValueInstance left, ValueInstance right)
	{
		if (left.IsList)
		{
			var items = new List<ValueInstance>(left.List.Items);
			var removeIndex = items.FindIndex(item => AreEqual(item, right));
			if (removeIndex >= 0)
				items.RemoveAt(removeIndex);
			return new ValueInstance(left.List.ReturnType, items);
		}
		return new ValueInstance(left.GetTypeExceptText(), left.Number - right.Number);
	}

	private (ValueInstance, ValueInstance) GetOperands(BinaryStatement statement) =>
		Memory.Registers.Count < 2
			? throw new OperandsRequired()
			: (Memory.Registers[statement.Registers[1]], Memory.Registers[statement.Registers[0]]);

	private void TryConditionalOperationExecution(BinaryStatement statement)
	{
		var (right, left) = GetOperands(statement);
		conditionFlag = statement.Instruction switch
		{
			Instruction.GreaterThan => left.Number > right.Number,
			Instruction.LessThan => left.Number < right.Number,
			Instruction.Equal => AreEqual(left, right),
			Instruction.NotEqual => !AreEqual(left, right),
			_ => false //ncrunch: no coverage
		};
	}

	public static bool AreEqual(ValueInstance value, ValueInstance other) => value.Equals(other);

	public static bool AreEqual(object? value, object? other)
	{
		if (ReferenceEquals(value, other))
			return true;
		if (value is ValueInstance valueInstance && other is ValueInstance otherValueInstance)
			return AreEqual(valueInstance, otherValueInstance);
		if (value is ValueInstance left)
			value = left.IsText
				? left.Text
				: left.Number;
		if (other is ValueInstance right)
			other = right.IsText
				? right.Text
				: right.Number;
		if (value is ValueListInstance valueList && other is ValueListInstance otherList)
			return valueList.Equals(otherList);
		if (value is ValueDictionaryInstance valueDictionary && other is ValueDictionaryInstance otherDictionary)
			return valueDictionary.Equals(otherDictionary);
		if (value is Expression leftExpression && other is Expression rightExpression)
			return leftExpression.Equals(rightExpression);
		return value?.Equals(other) ?? false;
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
			Memory.Registers[jumpIfNotZeroStatement.Register].Number > 0)
			instructionIndex += Convert.ToInt32(statement.Steps);
	}

	private void TryJumpToIdOperation(JumpToId statement)
	{
		if (!conditionFlag && statement.Instruction is Instruction.JumpToIdIfFalse ||
			conditionFlag && statement.Instruction is Instruction.JumpToIdIfTrue)
		{
			var id = statement.Id;
			var endIndex = statements.IndexOf(statements.First(jumpStatement =>
				jumpStatement.Instruction is Instruction.JumpEnd && jumpStatement is JumpToId jumpViaId &&
				jumpViaId.Id == id));
			if (endIndex != -1)
				instructionIndex = endIndex;
		}
	}

	public sealed class OperandsRequired : Exception { }
	private sealed class VariableNotFoundInMemory : Exception { }
}