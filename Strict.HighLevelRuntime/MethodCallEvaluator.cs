using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public sealed class MethodCallEvaluator(Interpreter interpreter)
{
	public ValueInstance EvaluateListCall(ListCall call, ExecutionContext ctx)
	{
		interpreter.Statistics.ListCallCount++;
		var listInstance = interpreter.RunExpression(call.List, ctx);
		var indexValue = interpreter.RunExpression(call.Index, ctx);
		if (listInstance.IsList || listInstance.IsText ||
			listInstance.TryGetValueTypeInstance()?.ReturnType.IsList == true)
			return listInstance.GetIteratorValue(interpreter.characterType, (int)indexValue.Number);
		throw new InvalidOperationException("List call needs a list, got: " + listInstance);
	}

	public ValueInstance Evaluate(MethodCall call, ExecutionContext ctx)
	{
		interpreter.Statistics.MethodCallCount++;
		var operatorType = GetOperatorCategory(call.Method.Name);
		if (operatorType != OperatorCategory.None)
			return EvaluateArithmeticOrCompareOrLogical(call, ctx, operatorType);
		var instance = call.Instance != null
			? interpreter.RunExpression(call.Instance, ctx)
			: call.Method.Name != Method.From
				? ctx.This
				: null;
		return ExecuteMethodCall(call, instance, ctx);
	}

	private enum OperatorCategory : byte
	{
		None,
		Arithmetic,
		Comparison,
		Logical
	}

	private static OperatorCategory GetOperatorCategory(string name) =>
		name switch
		{
			BinaryOperator.Plus or BinaryOperator.Minus or BinaryOperator.Multiply
				or BinaryOperator.Divide or BinaryOperator.Modulate or BinaryOperator.Power
				=> OperatorCategory.Arithmetic,
			BinaryOperator.Greater or BinaryOperator.Smaller or BinaryOperator.Is
				or BinaryOperator.GreaterOrEqual or BinaryOperator.SmallerOrEqual
				=> OperatorCategory.Comparison,
			BinaryOperator.And or BinaryOperator.Or or BinaryOperator.Xor or UnaryOperator.Not
				=> OperatorCategory.Logical,
			_ => OperatorCategory.None
		};

	private ValueInstance EvaluateArithmeticOrCompareOrLogical(MethodCall call,
		ExecutionContext ctx, OperatorCategory operatorType)
	{
		interpreter.Statistics.BinaryCount++;
		if (call.Instance == null || call.Arguments.Count != 1)
			throw new InvalidOperationException( //ncrunch: no coverage
				"Binary call must have instance and 1 argument");
		var leftInstance = interpreter.RunExpression(call.Instance, ctx);
		var rightInstance = interpreter.RunExpression(call.Arguments[0], ctx);
		return operatorType switch
		{
			OperatorCategory.Arithmetic => ExecuteArithmeticOperation(call, ctx, leftInstance, rightInstance),
			OperatorCategory.Comparison => ExecuteComparisonOperation(call, ctx, leftInstance, rightInstance),
			OperatorCategory.Logical => ExecuteLogicalBinaryOperation(call, ctx, leftInstance, rightInstance),
			_ => throw new InvalidOperationException("Unknown operator category") //ncrunch: no coverage
		};
	}

	private ValueInstance ExecuteArithmeticOperation(MethodCall call, ExecutionContext ctx,
		ValueInstance left, ValueInstance right)
	{
		interpreter.Statistics.ArithmeticCount++;
		var op = call.Method.Name;
		if (op == BinaryOperator.Plus && left.IsPrimitiveType(interpreter.characterType) &&
			right.IsPrimitiveType(interpreter.characterType))
			return new ValueInstance(left.ToExpressionCodeString() + right.ToExpressionCodeString());
		if (IsNumberLike(left) && IsNumberLike(right))
		{
			var l = left.Number;
			var r = right.Number;
			return op switch
			{
				BinaryOperator.Plus => new ValueInstance(interpreter.numberType, l + r),
				BinaryOperator.Minus => new ValueInstance(interpreter.numberType, l - r),
				BinaryOperator.Multiply => new ValueInstance(interpreter.numberType, l * r),
				BinaryOperator.Divide => new ValueInstance(interpreter.numberType, l / r),
				BinaryOperator.Modulate => new ValueInstance(interpreter.numberType, l % r),
				BinaryOperator.Power => new ValueInstance(interpreter.numberType, Math.Pow(l, r)),
				_ => ExecuteMethodCall(call, left, ctx) //ncrunch: no coverage
			};
		}
		if (left.IsText && right.IsText)
			return op == BinaryOperator.Plus
				? new ValueInstance(left.Text + right.Text)
				: throw new NotSupportedException("Only + operator is supported for Text, got: " + op);
		if (left.IsText && IsNumberLike(right))
		{
			return op == BinaryOperator.Plus
				? right.IsPrimitiveType(interpreter.characterType)
					?	new ValueInstance(left.Text + right.ToExpressionCodeString())
					: new ValueInstance(left.Text + right.Number)
				: throw new NotSupportedException("Only + operator is supported for Text+Number, got: " +
					op);
		}
		var leftList = ConvertToListValue(left);
		var rightList = ConvertToListValue(right);
		if (leftList.HasValue && rightList.HasValue)
		{
			if (op is BinaryOperator.Multiply or BinaryOperator.Divide &&
				leftList.Value.List.Items.Count != rightList.Value.List.Items.Count)
				return Error(ListsHaveDifferentDimensions, ctx, call);
			return op switch
			{
				BinaryOperator.Plus => CombineLists(leftList.Value, rightList.Value.List.Items, ctx, call),
				BinaryOperator.Minus => SubtractLists(leftList.Value, rightList.Value.List.Items),
				BinaryOperator.Multiply => MultiplyLists(leftList.Value.List.ReturnType, interpreter.numberType,
					leftList.Value.List.Items, rightList.Value.List.Items),
				BinaryOperator.Divide => DivideLists(leftList.Value.List.ReturnType, interpreter.numberType,
					leftList.Value.List.Items, rightList.Value.List.Items),
				_ => throw new NotSupportedException( //ncrunch: no coverage
					"Only +, -, *, / operators are supported for Lists, got: " + op)
			};
		}
		if (leftList.HasValue && right.IsPrimitiveType(interpreter.numberType))
		{
			if (op == BinaryOperator.Plus)
				return AddToList(leftList.Value, right);
			if (op == BinaryOperator.Minus)
				return RemoveFromList(leftList.Value, right);
			if (op == BinaryOperator.Multiply)
				return MultiplyList(leftList.Value.List.ReturnType, leftList.Value.List.Items, right.Number);
			if (op == BinaryOperator.Divide)
				return DivideList(leftList.Value.List.ReturnType, leftList.Value.List.Items, right.Number);
			throw new NotSupportedException( //ncrunch: no coverage
				"Only +, -, *, / operators are supported for List and Number, got: " + op);
		}
		if (IsCoreRuntimeType(call.Method.Type))
			throw new InvalidOperationException("Arithmetic fallback is not allowed for core type " +
				call.Method.Type.Name + " operator " + op + " with left=" + left + ", right=" + right);
		return ExecuteMethodCall(call, left, ctx); //ncrunch: no coverage
	}

	private static bool IsCoreRuntimeType(Type type) =>
		type.IsList || type.IsDictionary || type.IsNumber || type.IsText || type.IsBoolean ||
		type.IsCharacter || type.Name == Type.Range;

	private static ValueInstance? ConvertToListValue(ValueInstance value)
	{
		if (value.IsList)
			return value;
		var typeInstance = value.TryGetValueTypeInstance();
		if (typeInstance == null)
			return null;
		if (typeInstance.TryGetValue(Type.ElementsLowercase, out var elements))
			return elements.IsList
				? elements
				: ConvertToListValue(elements);
		if (typeInstance.TryGetValue(Type.IteratorLowercase, out var iterator))
			return iterator.IsList
				? iterator
				: ConvertToListValue(iterator);
		if (!typeInstance.ReturnType.IsList)
			return null;
		var length = value.GetIteratorLength();
		var iteratorValues = new ValueInstance[length];
		var listItemType = typeInstance.ReturnType.GetFirstImplementation();
		var characterType = listItemType.GetType(Type.Character);
		for (var index = 0; index < length; index++)
			iteratorValues[index] = value.GetIteratorValue(characterType, index);
		return new ValueInstance(typeInstance.ReturnType, iteratorValues);
	}

	private bool IsNumberLike(ValueInstance value) => value.IsNumberLike(interpreter.numberType);
	public const string ListsHaveDifferentDimensions = "listsHaveDifferentDimensions";

	private ValueInstance ExecuteComparisonOperation(MethodCall call, ExecutionContext ctx,
		ValueInstance left, ValueInstance right)
	{
		interpreter.Statistics.CompareCount++;
		var op = call.Method.Name;
		if (op is BinaryOperator.Is)
		{
			var rightInstance = right.TryGetValueTypeInstance();
			if (rightInstance is { ReturnType.IsError: true })
			{
				var leftInstance = left.TryGetValueTypeInstance();
				var matches = leftInstance != null && leftInstance.ReturnType.IsError &&
					leftInstance.ReturnType.IsSameOrCanBeUsedAs(rightInstance.ReturnType);
				return interpreter.ToBoolean(matches);
			}
			if (left.IsPrimitiveType(interpreter.characterType) && right.IsText)
				right = new ValueInstance(interpreter.characterType, right.Text[0]);
			if (left.IsText &&
				(right.IsPrimitiveType(interpreter.numberType) || right.IsPrimitiveType(interpreter.characterType)))
				right = new ValueInstance(right.ToExpressionCodeString());
			return interpreter.ToBoolean(left.Equals(right));
		}
		var l = left.Number;
		var r = right.Number;
		return op switch
		{
			BinaryOperator.Greater => interpreter.ToBoolean(l > r),
			BinaryOperator.Smaller => interpreter.ToBoolean(l < r),
			BinaryOperator.GreaterOrEqual => interpreter.ToBoolean(l >= r),
			BinaryOperator.SmallerOrEqual => interpreter.ToBoolean(l <= r),
			_ => ExecuteMethodCall(call, left, ctx) //ncrunch: no coverage
		};
	}

	private ValueInstance ExecuteLogicalBinaryOperation(MethodCall call, ExecutionContext ctx,
		ValueInstance left, ValueInstance right)
	{
		interpreter.Statistics.LogicalOperationCount++;
		return call.Method.Name switch
		{
			BinaryOperator.And => interpreter.ToBoolean(left.Boolean && right.Boolean),
			BinaryOperator.Or => interpreter.ToBoolean(left.Boolean || right.Boolean),
			BinaryOperator.Xor => interpreter.ToBoolean(left.Boolean ^ right.Boolean),
			_ => ExecuteMethodCall(call, left, ctx) //ncrunch: no coverage
		};
	}

	private ValueInstance CombineLists(ValueInstance leftList, List<ValueInstance> rightList,
		ExecutionContext ctx, MethodCall call)
	{
		var leftItemType = leftList.List.ReturnType.GetFirstImplementation();
		if (leftList.IsMutable)
		{
			foreach (var item in rightList)
				leftList.List.Items.Add(RightItemForCombineLists(leftItemType, item, ctx, call));
			return leftList;
		}
		var combined = new ValueInstance[leftList.List.Items.Count + rightList.Count];
		var itemIndex = 0;
		foreach (var item in leftList.List.Items)
			combined[itemIndex++] = item;
		foreach (var item in rightList)
			combined[itemIndex++] = RightItemForCombineLists(leftItemType, item, ctx, call);
		return new ValueInstance(leftList.List.ReturnType, combined);
	}

	private ValueInstance RightItemForCombineLists(Type leftItemType, ValueInstance item,
		ExecutionContext ctx, MethodCall call)
	{
		if (leftItemType.IsText && !item.IsText)
			return new ValueInstance(item.ToExpressionCodeString());
		if (leftItemType.IsNumber && (item.IsText || item.IsPrimitiveType(interpreter.characterType)))
			return double.TryParse(item.ToExpressionCodeString(), out var itemNumber)
				? new ValueInstance(leftItemType, itemNumber)
				: Error("Cannot downcast Text to Number for list: " + item, ctx, call);
		return item;
	}

	private static ValueInstance SubtractLists(ValueInstance leftList, List<ValueInstance> rightList)
	{
		if (leftList.IsMutable)
		{
			foreach (var rightItem in rightList)
			{
				var removeIndex = leftList.List.Items.FindIndex(leftItem => leftItem.Equals(rightItem));
				if (removeIndex >= 0)
					leftList.List.Items.RemoveAt(removeIndex);
			}
			return leftList;
		}
		var removed = new bool[rightList.Count];
		var temp = new ValueInstance[leftList.List.Items.Count];
		var itemCount = 0;
		for (var leftIndex = 0; leftIndex < leftList.List.Items.Count; leftIndex++)
		{
			var shouldKeep = true;
			for (var rightIndex = 0; rightIndex < rightList.Count; rightIndex++)
				if (!removed[rightIndex] && leftList.List.Items[leftIndex].Equals(rightList[rightIndex]))
				{
					removed[rightIndex] = true;
					shouldKeep = false;
					break;
				}
			if (shouldKeep)
				temp[itemCount++] = leftList.List.Items[leftIndex];
		}
		var result = new ValueInstance[itemCount];
		Array.Copy(temp, result, itemCount);
		return new ValueInstance(leftList.List.ReturnType, result);
	}

	private static ValueInstance AddToList(ValueInstance leftList, ValueInstance right)
	{
		var isLeftText = leftList.List.ReturnType is GenericTypeImplementation
		{
			Generic.Name: Type.List
		} list && list.ImplementationTypes[0].IsText;
		var rightItem = isLeftText && !right.IsText
			? new ValueInstance(right.ToExpressionCodeString())
			: right;
		if (leftList.IsMutable)
		{
			leftList.List.Items.Add(rightItem);
			return leftList;
		}
		var combined = new ValueInstance[leftList.List.Items.Count + 1];
		for (var itemIndex = 0; itemIndex < leftList.List.Items.Count; itemIndex++)
			combined[itemIndex] = leftList.List.Items[itemIndex];
		combined[leftList.List.Items.Count] = rightItem;
		return new ValueInstance(leftList.List.ReturnType, combined);
	}

	private static ValueInstance RemoveFromList(ValueInstance leftList, ValueInstance right)
	{
		if (leftList.IsMutable)
		{
			leftList.List.Items.RemoveAll(item => item.Equals(right));
			return leftList;
		}
		var count = 0;
		for (var index = 0; index < leftList.List.Items.Count; index++)
			if (!leftList.List.Items[index].Equals(right))
				count++;
		var result = new ValueInstance[count];
		var resultIndex = 0;
		for (var index = 0; index < leftList.List.Items.Count; index++)
			if (!leftList.List.Items[index].Equals(right))
				result[resultIndex++] = leftList.List.Items[index];
		return new ValueInstance(leftList.List.ReturnType, result);
	}

	private static ValueInstance MultiplyLists(Type leftListType, Type numberType,
		List<ValueInstance> leftList, List<ValueInstance> rightList)
	{
		var result = new ValueInstance[leftList.Count];
		for (var index = 0; index < leftList.Count; index++)
			result[index] =
				new ValueInstance(numberType, leftList[index].Number * rightList[index].Number);
		return new ValueInstance(leftListType, result);
	}

	private static ValueInstance DivideLists(Type leftListType, Type numberType,
		List<ValueInstance> leftList, List<ValueInstance> rightList)
	{
		var result = new ValueInstance[leftList.Count];
		for (var index = 0; index < leftList.Count; index++)
			result[index] =
				new ValueInstance(numberType, leftList[index].Number / rightList[index].Number);
		return new ValueInstance(leftListType, result);
	}

	private static ValueInstance MultiplyList(Type leftListType, List<ValueInstance> leftList,
		double rightNumber)
	{
		var result = new ValueInstance[leftList.Count];
		for (var i = 0; i < leftList.Count; i++)
			result[i] = new ValueInstance(leftList[i].GetType(), leftList[i].Number * rightNumber);
		return new ValueInstance(leftListType, result);
	}

	private static ValueInstance DivideList(Type leftListType, List<ValueInstance> leftList,
		double rightNumber)
	{
		var result = new ValueInstance[leftList.Count];
		for (var i = 0; i < leftList.Count; i++)
			result[i] = new ValueInstance(leftList[i].GetType(), leftList[i].Number / rightNumber);
		return new ValueInstance(leftListType, result);
	}

	private ValueInstance ExecuteMethodCall(MethodCall call, ValueInstance? instance,
		ExecutionContext ctx)
	{
		ValueInstance[] args;
		if (call.Arguments.Count == 0)
			args = [];
		else
		{
			args = new ValueInstance[call.Arguments.Count];
			for (var i = 0; i < call.Arguments.Count; i++)
				args[i] = interpreter.RunExpression(call.Arguments[i], ctx);
		}
		if (instance is { IsDictionary: true } && args.Length > 0 && call.Method.Name == "Add")
		{
			if (args.Length == 2)
				instance.Value.GetDictionaryItems()[args[0]] = args[1];
			return instance.Value;
		}
		var result = interpreter.Execute(call.Method, instance ?? interpreter.noneInstance, args, ctx);
		if (call.Method.ReturnType.IsMutable && call.Instance is VariableCall variableCall &&
			!instance.Equals(interpreter.noneInstance))
			ctx.Set(variableCall.Variable.Name, result);
		return result;
	}

	private ValueInstance Error(string name, ExecutionContext ctx, Expression? source = null)
	{
		var errorType = ctx.Method.GetType(Type.Error);
		var errorValues = new ValueInstance[errorType.Members.Count];
		for (var i = 0; i < errorType.Members.Count; i++)
			errorValues[i] = errorType.Members[i].Type.Name switch
			{
				nameof(Type.Name) or Type.Text => new ValueInstance(name),
				_ when errorType.Members[i].Type.IsList => CreateStacktrace(ctx, source),
				_ => throw new NotSupportedException( //ncrunch: no coverage
					"Error member not supported: " + errorType.Members[i])
			};
		return new ValueInstance(errorType, errorValues);
	}

	private ValueInstance CreateStacktrace(ExecutionContext ctx, Expression? source)
	{
		var stacktraceType = ctx.Method.GetType(Type.Stacktrace);
		var stackValues = new ValueInstance[stacktraceType.Members.Count];
		for (var i = 0; i < stacktraceType.Members.Count; i++)
			stackValues[i] = stacktraceType.Members[i].Type.Name switch
			{
				nameof(Method) => new ValueInstance(ctx.Method.GetType(nameof(Method)),
					CreateMethodValue(ctx.Method)),
				Type.Text or nameof(Type.Name) => new ValueInstance(ctx.Method.Type.FilePath),
				Type.Number => new ValueInstance(interpreter.numberType,
					source?.LineNumber ?? ctx.Method.TypeLineNumber),
				_ => throw new NotSupportedException( //ncrunch: no coverage
					"Stacktrace member not supported: " + stacktraceType.Members[i])
			};
		return new ValueInstance(interpreter.listType.GetGenericImplementation(stacktraceType),
			[new ValueInstance(stacktraceType, stackValues)]);
	}

	private static ValueInstance[] CreateMethodValue(Method method)
	{
		var methodType = method.GetType(nameof(Method));
		var values = new ValueInstance[methodType.Members.Count];
		for (var i = 0; i < methodType.Members.Count; i++)
			values[i] = methodType.Members[i].Type.Name switch
			{
				nameof(Type.Name) or Type.Text => new ValueInstance(method.Name),
				nameof(Type) => new ValueInstance(method.GetType(nameof(Type)),
					CreateTypeValue(method.Type)),
				_ => throw new NotSupportedException( //ncrunch: no coverage
					"Method member not supported: " + methodType.Members[i])
			};
		return values;
	}

	private static ValueInstance[] CreateTypeValue(Type type)
	{
		var typeType = type.GetType(nameof(Type));
		var values = new ValueInstance[typeType.Members.Count];
		for (var i = 0; i < typeType.Members.Count; i++)
			values[i] = typeType.Members[i].Type.Name switch
			{
				nameof(Type.Name) => new ValueInstance(type.Name),
				Type.Text => new ValueInstance(type.Package.FullName),
				_ => throw new NotSupportedException( //ncrunch: no coverage
					"Type member not supported: " + typeType.Members[i])
			};
		return values;
	}
}