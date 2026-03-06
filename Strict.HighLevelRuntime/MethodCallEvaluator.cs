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
		var op = call.Method.Name;
		if (IsArithmetic(op) || IsCompare(op) || IsLogical(op))
			return EvaluateArithmeticOrCompareOrLogical(call, ctx);
		var instance = call.Instance != null
			? executor.RunExpression(call.Instance, ctx)
			: call.Method.Name != Method.From
				? ctx.This
				: null;
		return ExecuteMethodCall(call, instance, ctx);
	}

	private static bool IsArithmetic(string name) =>
		name is BinaryOperator.Plus or BinaryOperator.Minus or BinaryOperator.Multiply
			or BinaryOperator.Divide or BinaryOperator.Modulate or BinaryOperator.Power;

	private static bool IsCompare(string name) =>
		name is BinaryOperator.Greater or BinaryOperator.Smaller or BinaryOperator.Is
			or BinaryOperator.GreaterOrEqual or BinaryOperator.SmallerOrEqual or UnaryOperator.Not;

	private static bool IsLogical(string name) =>
		name is BinaryOperator.And or BinaryOperator.Or or BinaryOperator.Xor or UnaryOperator.Not;

	private ValueInstance EvaluateArithmeticOrCompareOrLogical(MethodCall call,
		ExecutionContext ctx)
	{
		executor.Statistics.BinaryCount++;
		if (call.Instance == null || call.Arguments.Count != 1)
			throw new InvalidOperationException("Binary call must have instance and 1 argument"); //ncrunch: no coverage
		var leftInstance = executor.RunExpression(call.Instance, ctx);
		var rightInstance = executor.RunExpression(call.Arguments[0], ctx);
		return IsArithmetic(call.Method.Name)
			? ExecuteArithmeticOperation(call, ctx, leftInstance, rightInstance)
			: IsCompare(call.Method.Name)
				? ExecuteComparisonOperation(call, ctx, leftInstance, rightInstance)
				: ExecuteLogicalBinaryOperation(call, ctx, leftInstance, rightInstance);
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
				left.List.Items.Count != right.List.Items.Count)
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
		if (op is BinaryOperator.Is or UnaryOperator.Not)
		{
			var rightInstance = right.TryGetValueTypeInstance();
			if (rightInstance is { ReturnType.IsError: true })
			{
				var leftInstance = left.TryGetValueTypeInstance();
				var matches = leftInstance != null && leftInstance.ReturnType.IsError &&
					leftInstance.ReturnType.IsSameOrCanBeUsedAs(rightInstance.ReturnType);
				if (op is not BinaryOperator.Is)
					matches = !matches;
				return executor.ToBoolean(matches);
			}
			if (left.IsPrimitiveType(executor.characterType) && right.IsText)
				right = new ValueInstance(executor.characterType, right.Text[0]);
			if (left.IsText &&
				(right.IsPrimitiveType(executor.numberType) || right.IsPrimitiveType(executor.characterType)))
				right = new ValueInstance(right.ToExpressionCodeString());
			var equals = left.Equals(right);
			if (op is not BinaryOperator.Is)
				equals = !equals;
			return executor.ToBoolean(equals);
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
		var combined = new List<ValueInstance>(leftList.Count + rightList.Count);
		var isLeftText = listType is GenericTypeImplementation { Generic.Name: Type.List } list &&
			list.ImplementationTypes[0].IsText;
		foreach (var item in leftList)
			combined.Add(item);
		foreach (var item in rightList)
			combined.Add(isLeftText && !item.IsText
				? new ValueInstance(item.ToExpressionCodeString())
				: item);
		return new ValueInstance(listType, combined);
	}

	private static ValueInstance SubtractLists(Type listType, IReadOnlyList<ValueInstance> leftList,
		IReadOnlyList<ValueInstance> rightList)
	{
		var remainder = new List<ValueInstance>();
		foreach (var item in leftList)
			remainder.Add(item);
		foreach (var item in rightList)
			remainder.Remove(item);
		return new ValueInstance(listType, remainder);
	}

	private static ValueInstance MultiplyLists(Type leftListType, Type numberType,
		IReadOnlyList<ValueInstance> leftList, IReadOnlyList<ValueInstance> rightList)
	{
		var result = new List<ValueInstance>();
		for (var index = 0; index < leftList.Count; index++)
			result.Add(new ValueInstance(numberType, leftList[index].Number * rightList[index].Number));
		return new ValueInstance(leftListType, result);
	}

	private static ValueInstance DivideLists(Type leftListType, Type numberType,
		IReadOnlyList<ValueInstance> leftList, IReadOnlyList<ValueInstance> rightList)
	{
		var result = new List<ValueInstance>();
		for (var index = 0; index < leftList.Count; index++)
			result.Add(new ValueInstance(numberType, leftList[index].Number / rightList[index].Number));
		return new ValueInstance(leftListType, result);
	}

	private static ValueInstance AddToList(Type leftListType, IReadOnlyList<ValueInstance> leftList,
		ValueInstance right)
	{
		var combined = new List<ValueInstance>(leftList.Count + 1);
		var isLeftText = leftListType is GenericTypeImplementation { Generic.Name: Type.List } list &&
			list.ImplementationTypes[0].IsText;
		foreach (var item in leftList)
			combined.Add(item);
		combined.Add(isLeftText && !right.IsText
			? new ValueInstance(right.ToExpressionCodeString())
			: right);
		return new ValueInstance(leftListType, combined);
	}

	private static ValueInstance RemoveFromList(Type leftListType,
		IReadOnlyList<ValueInstance> leftList, ValueInstance right)
	{
		var result = new List<ValueInstance>();
		foreach (var item in leftList)
			if (!item.Equals(right))
				result.Add(item);
		return new ValueInstance(leftListType, result);
	}

	private static ValueInstance MultiplyList(Type leftListType,
		IReadOnlyList<ValueInstance> leftList, double rightNumber)
	{
		var result = new List<ValueInstance>(leftList.Count);
		foreach (var item in leftList)
			result.Add(new ValueInstance(item.GetTypeExceptText(), item.Number * rightNumber));
		return new ValueInstance(leftListType, result);
	}

	private static ValueInstance DivideList(Type leftListType, IReadOnlyList<ValueInstance> leftList,
		double rightNumber)
	{
		var result = new List<ValueInstance>(leftList.Count);
		foreach (var item in leftList)
			result.Add(new ValueInstance(item.GetTypeExceptText(), item.Number / rightNumber));
		return new ValueInstance(leftListType, result);
	}

	private ValueInstance ExecuteMethodCall(MethodCall call, ValueInstance? instance,
		ExecutionContext ctx)
	{
		IReadOnlyList<ValueInstance> args;
		if (call.Arguments.Count == 0)
			args = [];
		else
		{
			var argsArray = new ValueInstance[call.Arguments.Count];
			for (var i = 0; i < call.Arguments.Count; i++)
				argsArray[i] = executor.RunExpression(call.Arguments[i], ctx);
			args = argsArray;
		}
		if (instance is { IsDictionary: true } && args.Count > 0 && call.Method.Name == "Add")
		{
			if (args.Count == 2)
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
		var errorMembers = new Dictionary<string, ValueInstance>(StringComparer.OrdinalIgnoreCase);
		var errorType = ctx.Method.GetType(Type.Error);
		foreach (var member in errorType.Members)
			errorMembers[member.Name] = member.Type.Name switch
			{
				nameof(Type.Name) or Type.Text => new ValueInstance(name),
				_ when member.Type.IsList => CreateStacktrace(ctx, source),
				_ => throw new NotSupportedException("Error member not supported: " + member) //ncrunch: no coverage
			};
		return new ValueInstance(errorType, errorMembers);
	}

	private ValueInstance CreateStacktrace(ExecutionContext ctx, Expression? source)
	{
		var members = new Dictionary<string, ValueInstance>(StringComparer.OrdinalIgnoreCase);
		var stacktraceType = ctx.Method.GetType(Type.Stacktrace);
		foreach (var member in stacktraceType.Members)
			members[member.Name] = member.Type.Name switch
			{
				nameof(Method) => new ValueInstance(ctx.Method.GetType(nameof(Method)),
					CreateMethodValue(ctx.Method)),
				Type.Text or nameof(Type.Name) => new ValueInstance(ctx.Method.Type.FilePath),
				Type.Number => new ValueInstance(executor.numberType,
					source?.LineNumber ?? ctx.Method.TypeLineNumber),
				_ => throw new NotSupportedException("Stacktrace member not supported: " + member) //ncrunch: no coverage
			};
		return new ValueInstance(executor.listType.GetGenericImplementation(stacktraceType),
			[new ValueInstance(stacktraceType, members)]);
	}

	private static Dictionary<string, ValueInstance> CreateMethodValue(Method method)
	{
		var values = new Dictionary<string, ValueInstance>(StringComparer.OrdinalIgnoreCase);
		var methodType = method.GetType(nameof(Method));
		foreach (var member in methodType.Members)
			values[member.Name] = member.Type.Name switch
			{
				nameof(Type.Name) or Type.Text => new ValueInstance(method.Name),
				nameof(Type) => new ValueInstance(method.GetType(nameof(Type)),
					CreateTypeValue(method.Type)),
				_ => throw new NotSupportedException("Method member not supported: " + member) //ncrunch: no coverage
			};
		return values;
	}

	private static Dictionary<string, ValueInstance> CreateTypeValue(Type type)
	{
		var values = new Dictionary<string, ValueInstance>(StringComparer.OrdinalIgnoreCase);
		var typeType = type.GetType(nameof(Type));
		foreach (var member in typeType.Members)
			values[member.Name] = member.Type.Name switch
			{
				nameof(Type.Name) => new ValueInstance(type.Name),
				Type.Text => new ValueInstance(type.Package.FullName),
				_ => throw new NotSupportedException("Type member not supported: " + member) //ncrunch: no coverage
			};
		return values;
	}
}