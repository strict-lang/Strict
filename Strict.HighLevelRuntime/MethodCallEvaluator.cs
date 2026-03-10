using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public sealed class MethodCallEvaluator(Executor executor)
{
	public ValueInstance EvaluateListCall(ListCall call, ExecutionContext ctx)
	{
		executor.Statistics.ListCallCount++;
		var listInstance = executor.RunExpression(call.List, ctx);
		var indexValue = executor.RunExpression(call.Index, ctx);
		return listInstance.IsList
			? listInstance.GetIteratorValue(executor.characterType, (int)indexValue.Number)
			: throw new InvalidOperationException("List call needs a list, got: " + listInstance);
	}

	public ValueInstance Evaluate(MethodCall call, ExecutionContext ctx)
	{
		executor.Statistics.MethodCallCount++;
		var operatorType = GetOperatorCategory(call.Method.Name);
		if (operatorType != OperatorCategory.None)
			return EvaluateArithmeticOrCompareOrLogical(call, ctx, operatorType);
		var instance = call.Instance != null
			? executor.RunExpression(call.Instance, ctx)
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
		executor.Statistics.BinaryCount++;
		if (call.Instance == null || call.Arguments.Count != 1)
			throw new InvalidOperationException("Binary call must have instance and 1 argument"); //ncrunch: no coverage
		var leftInstance = executor.RunExpression(call.Instance, ctx);
		var rightInstance = executor.RunExpression(call.Arguments[0], ctx);
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
		executor.Statistics.ArithmeticCount++;
		var op = call.Method.Name;
		if (IsNumberLike(left) && IsNumberLike(right))
		{
			var l = left.Number;
			var r = right.Number;
			return op switch
			{
				BinaryOperator.Plus => new ValueInstance(executor.numberType, l + r),
				BinaryOperator.Minus => new ValueInstance(executor.numberType, l - r),
				BinaryOperator.Multiply => new ValueInstance(executor.numberType, l * r),
				BinaryOperator.Divide => new ValueInstance(executor.numberType, l / r),
				BinaryOperator.Modulate => new ValueInstance(executor.numberType, l % r),
				BinaryOperator.Power => new ValueInstance(executor.numberType, Math.Pow(l, r)),
				_ => ExecuteMethodCall(call, left, ctx) //ncrunch: no coverage
			};
		}
		if (left.IsText && right.IsText)
			return op == BinaryOperator.Plus
				? new ValueInstance(left.Text + right.Text)
				: throw new NotSupportedException("Only + operator is supported for Text, got: " + op);
		if (left.IsText && right.IsPrimitiveType(executor.numberType))
		{
			return op == BinaryOperator.Plus
				? new ValueInstance(left.Text + right.Number)
				: throw new NotSupportedException("Only + operator is supported for Text+Number, got: " +
					op);
		}
		if (left.IsList && right.IsList)
		{
			if (op is BinaryOperator.Multiply or BinaryOperator.Divide &&
				left.List.Items.Length != right.List.Items.Length)
				return Error(ListsHaveDifferentDimensions, ctx, call);
			return op switch
			{
				BinaryOperator.Plus => CombineLists(left.List.ReturnType, left.List.Items,
					right.List.Items),
				BinaryOperator.Minus => SubtractLists(left.List.ReturnType, left.List.Items,
					right.List.Items),
				BinaryOperator.Multiply => MultiplyLists(left.List.ReturnType, executor.numberType,
					left.List.Items, right.List.Items),
				BinaryOperator.Divide => DivideLists(left.List.ReturnType, executor.numberType,
					left.List.Items, right.List.Items),
				_ => throw new NotSupportedException( //ncrunch: no coverage
					"Only +, -, *, / operators are supported for Lists, got: " + op)
			};
		}
		if (left.IsList && right.IsPrimitiveType(executor.numberType))
		{
			if (op == BinaryOperator.Plus)
				return AddToList(left.List.ReturnType, left.List.Items, right);
			if (op == BinaryOperator.Minus)
				return RemoveFromList(left.List.ReturnType, left.List.Items, right);
			if (op == BinaryOperator.Multiply)
				return MultiplyList(left.List.ReturnType, left.List.Items, right.Number);
			if (op == BinaryOperator.Divide)
				return DivideList(left.List.ReturnType, left.List.Items, right.Number);
			throw new NotSupportedException( //ncrunch: no coverage
				"Only +, -, *, / operators are supported for List and Number, got: " + op);
		}
		return ExecuteMethodCall(call, left, ctx); //ncrunch: no coverage
	}

	private bool IsNumberLike(ValueInstance value) => value.IsNumberLike(executor.numberType);
	public const string ListsHaveDifferentDimensions = "listsHaveDifferentDimensions";

	private ValueInstance ExecuteComparisonOperation(MethodCall call, ExecutionContext ctx,
		ValueInstance left, ValueInstance right)
	{
		executor.Statistics.CompareCount++;
		var op = call.Method.Name;
		if (op is BinaryOperator.Is)
		{
			var rightInstance = right.TryGetValueTypeInstance();
			if (rightInstance is { ReturnType.IsError: true })
			{
				var leftInstance = left.TryGetValueTypeInstance();
				var matches = leftInstance != null && leftInstance.ReturnType.IsError &&
					leftInstance.ReturnType.IsSameOrCanBeUsedAs(rightInstance.ReturnType);
				return executor.ToBoolean(matches);
			}
			if (left.IsPrimitiveType(executor.characterType) && right.IsText)
				right = new ValueInstance(executor.characterType, right.Text[0]);
			if (left.IsText &&
				(right.IsPrimitiveType(executor.numberType) || right.IsPrimitiveType(executor.characterType)))
				right = new ValueInstance(right.ToExpressionCodeString());
			return executor.ToBoolean(left.Equals(right));
		}
		var l = left.Number;
		var r = right.Number;
		return op switch
		{
			BinaryOperator.Greater => executor.ToBoolean(l > r),
			BinaryOperator.Smaller => executor.ToBoolean(l < r),
			BinaryOperator.GreaterOrEqual => executor.ToBoolean(l >= r),
			BinaryOperator.SmallerOrEqual => executor.ToBoolean(l <= r),
			_ => ExecuteMethodCall(call, left, ctx) //ncrunch: no coverage
		};
	}

	private ValueInstance ExecuteLogicalBinaryOperation(MethodCall call, ExecutionContext ctx,
		ValueInstance left, ValueInstance right)
	{
		executor.Statistics.LogicalOperationCount++;
		return call.Method.Name switch
		{
			BinaryOperator.And => executor.ToBoolean(left.Boolean && right.Boolean),
			BinaryOperator.Or => executor.ToBoolean(left.Boolean || right.Boolean),
			BinaryOperator.Xor => executor.ToBoolean(left.Boolean ^ right.Boolean),
			_ => ExecuteMethodCall(call, left, ctx) //ncrunch: no coverage
		};
	}

	private static ValueInstance CombineLists(Type listType, IReadOnlyList<ValueInstance> leftList,
		IReadOnlyList<ValueInstance> rightList)
	{
		var isLeftText = listType is GenericTypeImplementation { Generic.Name: Type.List } list &&
			list.ImplementationTypes[0].IsText;
		var combined = new ValueInstance[leftList.Count + rightList.Count];
		var idx = 0;
		for (var i = 0; i < leftList.Count; i++)
			combined[idx++] = leftList[i];
		for (var i = 0; i < rightList.Count; i++)
			combined[idx++] = isLeftText && !rightList[i].IsText
				? new ValueInstance(rightList[i].ToExpressionCodeString())
				: rightList[i];
		return new ValueInstance(listType, combined);
	}

	private static ValueInstance SubtractLists(Type listType, IReadOnlyList<ValueInstance> leftList,
		IReadOnlyList<ValueInstance> rightList)
	{
		var removed = new bool[rightList.Count];
		var temp = new ValueInstance[leftList.Count];
		var count = 0;
		for (var i = 0; i < leftList.Count; i++)
		{
			var keep = true;
			for (var j = 0; j < rightList.Count; j++)
				if (!removed[j] && leftList[i].Equals(rightList[j]))
				{
					removed[j] = true;
					keep = false;
					break;
				}
			if (keep)
				temp[count++] = leftList[i];
		}
		var result = new ValueInstance[count];
		Array.Copy(temp, result, count);
		return new ValueInstance(listType, result);
	}

	private static ValueInstance MultiplyLists(Type leftListType, Type numberType,
		IReadOnlyList<ValueInstance> leftList, IReadOnlyList<ValueInstance> rightList)
	{
		var result = new ValueInstance[leftList.Count];
		for (var index = 0; index < leftList.Count; index++)
			result[index] = new ValueInstance(numberType, leftList[index].Number * rightList[index].Number);
		return new ValueInstance(leftListType, result);
	}

	private static ValueInstance DivideLists(Type leftListType, Type numberType,
		IReadOnlyList<ValueInstance> leftList, IReadOnlyList<ValueInstance> rightList)
	{
		var result = new ValueInstance[leftList.Count];
		for (var index = 0; index < leftList.Count; index++)
			result[index] = new ValueInstance(numberType, leftList[index].Number / rightList[index].Number);
		return new ValueInstance(leftListType, result);
	}

	private static ValueInstance AddToList(Type leftListType, IReadOnlyList<ValueInstance> leftList,
		ValueInstance right)
	{
		var isLeftText = leftListType is GenericTypeImplementation { Generic.Name: Type.List } list &&
			list.ImplementationTypes[0].IsText;
		var combined = new ValueInstance[leftList.Count + 1];
		for (var i = 0; i < leftList.Count; i++)
			combined[i] = leftList[i];
		combined[leftList.Count] = isLeftText && !right.IsText
			? new ValueInstance(right.ToExpressionCodeString())
			: right;
		return new ValueInstance(leftListType, combined);
	}

	private static ValueInstance RemoveFromList(Type leftListType,
		IReadOnlyList<ValueInstance> leftList, ValueInstance right)
	{
		var count = 0;
		for (var i = 0; i < leftList.Count; i++)
			if (!leftList[i].Equals(right))
				count++;
		var result = new ValueInstance[count];
		var idx = 0;
		for (var i = 0; i < leftList.Count; i++)
			if (!leftList[i].Equals(right))
				result[idx++] = leftList[i];
		return new ValueInstance(leftListType, result);
	}

	private static ValueInstance MultiplyList(Type leftListType,
		IReadOnlyList<ValueInstance> leftList, double rightNumber)
	{
		var result = new ValueInstance[leftList.Count];
		for (var i = 0; i < leftList.Count; i++)
			result[i] = new ValueInstance(leftList[i].GetType(), leftList[i].Number * rightNumber);
		return new ValueInstance(leftListType, result);
	}

	private static ValueInstance DivideList(Type leftListType, IReadOnlyList<ValueInstance> leftList,
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
				args[i] = executor.RunExpression(call.Arguments[i], ctx);
		}
		if (instance is { IsDictionary: true } && args.Length > 0 && call.Method.Name == "Add")
		{
			if (args.Length == 2)
				instance.Value.GetDictionaryItems()[args[0]] = args[1];
			return instance.Value;
		}
		var result = executor.Execute(call.Method, instance ?? executor.noneInstance, args, ctx);
		if (call.Method.ReturnType.IsMutable && call.Instance is VariableCall variableCall &&
			!instance.Equals(executor.noneInstance))
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
				_ => throw new NotSupportedException("Error member not supported: " + errorType.Members[i]) //ncrunch: no coverage
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
				Type.Number => new ValueInstance(executor.numberType,
					source?.LineNumber ?? ctx.Method.TypeLineNumber),
				_ => throw new NotSupportedException("Stacktrace member not supported: " + stacktraceType.Members[i]) //ncrunch: no coverage
			};
		return new ValueInstance(executor.listType.GetGenericImplementation(stacktraceType),
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
				_ => throw new NotSupportedException("Method member not supported: " + methodType.Members[i]) //ncrunch: no coverage
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
				_ => throw new NotSupportedException("Type member not supported: " + typeType.Members[i]) //ncrunch: no coverage
			};
		return values;
	}
}