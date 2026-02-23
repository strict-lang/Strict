using Strict.Expressions;
using Strict.Language;
using System.Collections;
using System.Globalization;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public sealed class MethodCallEvaluator(Executor executor)
{
	public ValueInstance EvaluateListCall(ListCall call, ExecutionContext ctx)
	{
		var listInstance = executor.RunExpression(call.List, ctx);
		var indexValue = executor.RunExpression(call.Index, ctx);
		var index = Convert.ToInt32(EqualsExtensions.NumberToDouble(indexValue.Value));
		if (listInstance.Value is IList list)
			return list[index] as ValueInstance ?? new ValueInstance(call.ReturnType, list[index]);
		throw new InvalidOperationException("List call can only be used on iterators, got: " + //ncrunch: no coverage
			listInstance);
	}

	public ValueInstance Evaluate(MethodCall call, ExecutionContext ctx)
	{
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
		if (call.Instance == null || call.Arguments.Count != 1)
			throw new InvalidOperationException("Binary call must have instance and 1 argument"); //ncrunch: no coverage
		var leftInstance = executor.RunExpression(call.Instance, ctx);
		var rightInstance = executor.RunExpression(call.Arguments[0], ctx);
		return IsArithmetic(call.Method.Name)
			? ExecuteArithmeticOperation(call, ctx, leftInstance, rightInstance)
			: IsCompare(call.Method.Name)
				? ExecuteComparisonOperation(call, ctx, leftInstance, rightInstance)
				: ExecuteBinaryOperation(call, ctx, leftInstance, rightInstance);
	}

	private ValueInstance ExecuteArithmeticOperation(MethodCall call, ExecutionContext ctx,
		ValueInstance leftInstance, ValueInstance rightInstance)
	{
		var op = call.Method.Name;
		var left = leftInstance.Value;
		var right = rightInstance.Value;
		if (leftInstance.ReturnType.Name == Base.Number &&
			rightInstance.ReturnType.Name == Base.Number)
		{
			var l = EqualsExtensions.NumberToDouble(left);
			var r = EqualsExtensions.NumberToDouble(right);
			return op switch
			{
				BinaryOperator.Plus => Number(call.Method, l + r),
				BinaryOperator.Minus => Number(call.Method, l - r),
				BinaryOperator.Multiply => Number(call.Method, l * r),
				BinaryOperator.Divide => Number(call.Method, l / r),
				BinaryOperator.Modulate => Number(call.Method, l % r),
				BinaryOperator.Power => Number(call.Method, Math.Pow(l, r)),
				_ => ExecuteMethodCall(call, leftInstance, ctx) //ncrunch: no coverage
			};
		}
		if (leftInstance.ReturnType.Name == Base.Text && rightInstance.ReturnType.Name == Base.Text)
		{
			return op == BinaryOperator.Plus
				? new ValueInstance(leftInstance.ReturnType, (string)left! + (string)right!)
				: throw new NotSupportedException("Only + operator is supported for Text, got: " + op);
		}
		if (leftInstance.ReturnType.Name == Base.Text && rightInstance.ReturnType.Name == Base.Number)
		{
			return op == BinaryOperator.Plus
				? new ValueInstance(leftInstance.ReturnType,
					(string)left! + (int)EqualsExtensions.NumberToDouble(right))
				: throw new NotSupportedException("Only + operator is supported for Text+Number, got: " + op);
		}
		if (leftInstance.ReturnType.IsIterator && rightInstance.ReturnType.IsIterator)
		{
			if (left is not IList<ValueInstance> leftList ||
				right is not IList<ValueInstance> rightList)
				throw new InvalidOperationException( //ncrunch: no coverage
					"Expected List<ValueInstance> for iterator operation, " +
					"other iterators are not yet supported: left=" + left + ", right=" + right);
			if (op is BinaryOperator.Multiply or BinaryOperator.Divide &&
				leftList.Count != rightList.Count)
				return Error(ListsHaveDifferentDimensions, ctx, call);
			return op switch
			{
				BinaryOperator.Plus => CombineLists(leftInstance.ReturnType, leftList, rightList),
				BinaryOperator.Minus => SubtractLists(leftInstance.ReturnType, leftList, rightList),
				BinaryOperator.Multiply => MultiplyLists(leftInstance.ReturnType, leftList, rightList),
				BinaryOperator.Divide => DivideLists(leftInstance.ReturnType, leftList, rightList),
				_ => throw new NotSupportedException( //ncrunch: no coverage
					"Only +, -, *, / operators are supported for Lists, got: " + op)
			};
		}
		if (leftInstance.ReturnType.IsIterator && rightInstance.ReturnType.Name == Base.Number)
		{
			if (left is not IList<ValueInstance> leftList)
				throw new InvalidOperationException("Expected left list for iterator operation " + //ncrunch: no coverage
					op + ": left=" + left + ", right=" + right);
			if (op == BinaryOperator.Plus)
				return AddToList(leftInstance.ReturnType, leftList, rightInstance);
			if (op == BinaryOperator.Minus)
				return RemoveFromList(leftInstance.ReturnType, leftList, rightInstance);
			if (right is not double rightNumber)
				throw new InvalidOperationException("Expected right number for iterator operation " + //ncrunch: no coverage
					op + ": left=" + left + ", right=" + right);
			if (op == BinaryOperator.Multiply)
				return MultiplyList(leftInstance.ReturnType, leftList, rightNumber);
			if (op == BinaryOperator.Divide)
				return DivideList(leftInstance.ReturnType, leftList, rightNumber);
			throw new NotSupportedException( //ncrunch: no coverage
				"Only +, -, *, / operators are supported for List and Number, got: " + op);
		}
		return ExecuteMethodCall(call, leftInstance, ctx); //ncrunch: no coverage
	}

	private static ValueInstance Number(Context any, double n) => new(any.GetType(Base.Number), n);
	public const string ListsHaveDifferentDimensions = "listsHaveDifferentDimensions";

	private ValueInstance ExecuteComparisonOperation(MethodCall call, ExecutionContext ctx,
		ValueInstance leftInstance, ValueInstance rightInstance)
	{
		var op = call.Method.Name;
		var left = leftInstance.Value!;
		var right = rightInstance.Value!;
		if (op is BinaryOperator.Is or UnaryOperator.Not)
		{
			if (rightInstance.ReturnType.IsError)
			{
				var matches = rightInstance.ReturnType.Name == Base.Error
					? leftInstance.ReturnType.IsError
					: leftInstance.ReturnType.IsSameOrCanBeUsedAs(rightInstance.ReturnType);
				return op is BinaryOperator.Is
					? Executor.Bool(call.Method, matches)
					: Executor.Bool(call.Method, !matches);
			}
			if (leftInstance.ReturnType.Name == Base.Character && right is string rightText)
			{
				right = (int)rightText[0];
				rightInstance = new ValueInstance(leftInstance.ReturnType, right);
			}
			if (leftInstance.ReturnType.Name == Base.Text && right is int rightInt)
			{
				right = rightInt + "";
				rightInstance = new ValueInstance(leftInstance.ReturnType, right);
			}
			var equals = leftInstance.Equals(rightInstance);
			return Executor.Bool(call.Method, op is BinaryOperator.Is
				? equals
				: !equals);
		}
		var l = EqualsExtensions.NumberToDouble(left);
		var r = EqualsExtensions.NumberToDouble(right);
		return op switch
		{
			BinaryOperator.Greater => Executor.Bool(call.Method, l > r),
			BinaryOperator.Smaller => Executor.Bool(call.Method, l < r),
			BinaryOperator.GreaterOrEqual => Executor.Bool(call.Method, l >= r),
			BinaryOperator.SmallerOrEqual => Executor.Bool(call.Method, l <= r),
			_ => ExecuteMethodCall(call, leftInstance, ctx) //ncrunch: no coverage
		};
	}

	private ValueInstance ExecuteBinaryOperation(MethodCall call, ExecutionContext ctx,
		ValueInstance leftInstance, ValueInstance rightInstance)
	{
		var left = leftInstance.Value;
		var right = rightInstance.Value;
		return call.Method.Name switch
		{
			BinaryOperator.And => Executor.Bool(call.Method, Executor.ToBool(left) && Executor.ToBool(right)),
			BinaryOperator.Or => Executor.Bool(call.Method, Executor.ToBool(left) || Executor.ToBool(right)),
			BinaryOperator.Xor => Executor.Bool(call.Method, Executor.ToBool(left) ^ Executor.ToBool(right)),
			_ => ExecuteMethodCall(call, leftInstance, ctx) //ncrunch: no coverage
		};
	}

	private static ValueInstance CombineLists(Type listType, ICollection<ValueInstance> leftList,
		ICollection<ValueInstance> rightList)
	{
		var combined = new List<ValueInstance>(leftList.Count + rightList.Count);
		var isLeftText = listType is GenericTypeImplementation { Generic.Name: Base.List } list &&
			list.ImplementationTypes[0].Name == Base.Text;
		foreach (var item in leftList)
			combined.Add(item);
		foreach (var item in rightList)
			combined.Add(isLeftText && item.ReturnType.Name != Base.Text
				? new ValueInstance(listType.GetType(Base.Text), item.Value?.ToString())
				: item);
		return new ValueInstance(listType, combined);
	}

	private static ValueInstance SubtractLists(Type listType, IEnumerable<ValueInstance> leftList,
		IEnumerable<ValueInstance> rightList)
	{
		var remainder = new List<ValueInstance>();
		foreach (var item in leftList)
			remainder.Add(item);
		foreach (var item in rightList)
			remainder.Remove(item);
		return new ValueInstance(listType, remainder);
	}

	private static ValueInstance MultiplyLists(Type leftListType, IList<ValueInstance> leftList,
		IList<ValueInstance> rightList)
	{
		var result = new List<ValueInstance>();
		for (var index = 0; index < leftList.Count; index++)
			result.Add(new ValueInstance(leftListType.GetType(Base.Number),
				EqualsExtensions.NumberToDouble(leftList[index].Value) *
				EqualsExtensions.NumberToDouble(rightList[index].Value)));
		return new ValueInstance(leftListType, result);
	}

	private static ValueInstance DivideLists(Type leftListType, IList<ValueInstance> leftList,
		IList<ValueInstance> rightList)
	{
		var result = new List<ValueInstance>();
		for (var index = 0; index < leftList.Count; index++)
			result.Add(new ValueInstance(leftListType.GetType(Base.Number),
				EqualsExtensions.NumberToDouble(leftList[index].Value) /
				EqualsExtensions.NumberToDouble(rightList[index].Value)));
		return new ValueInstance(leftListType, result);
	}

	private static ValueInstance AddToList(Type leftListType, ICollection<ValueInstance> leftList,
		ValueInstance right)
	{
		var combined = new List<ValueInstance>(leftList.Count + 1);
		var isLeftText = leftListType is GenericTypeImplementation { Generic.Name: Base.List } list &&
			list.ImplementationTypes[0].Name == Base.Text;
		foreach (var item in leftList)
			combined.Add(item);
		combined.Add(isLeftText && right.ReturnType.Name != Base.Text
			? new ValueInstance(leftListType.GetType(Base.Text), ConvertToText(right.Value))
			: right);
		return new ValueInstance(leftListType, combined);
	}

	private static string ConvertToText(object? value) =>
		value switch
		{
			string text => text, //ncrunch: no coverage
			double number => number.ToString(CultureInfo.InvariantCulture),
			int number => number.ToString(CultureInfo.InvariantCulture), //ncrunch: no coverage
			_ => value?.ToString() ?? string.Empty //ncrunch: no coverage
		};

	private static ValueInstance RemoveFromList(Type leftListType,
		IEnumerable<ValueInstance> leftList, ValueInstance right)
	{
		var result = new List<ValueInstance>();
		foreach (var item in leftList)
			if (!item.Equals(right))
				result.Add(item);
		return new ValueInstance(leftListType, result);
	}

	private static ValueInstance MultiplyList(Type leftListType,
		ICollection<ValueInstance> leftList, double rightNumber)
	{
		var result = new List<ValueInstance>(leftList.Count);
		foreach (var item in leftList)
			result.Add(new ValueInstance(item.ReturnType,
				EqualsExtensions.NumberToDouble(item.Value) * rightNumber));
		return new ValueInstance(leftListType, result);
	}

	private static ValueInstance DivideList(Type leftListType, ICollection<ValueInstance> leftList,
		double rightNumber)
	{
		var result = new List<object?>(leftList.Count);
		foreach (var item in leftList)
			result.Add(new ValueInstance(item.ReturnType,
				EqualsExtensions.NumberToDouble(item.Value) / rightNumber));
		return new ValueInstance(leftListType, result);
	}

	private ValueInstance ExecuteMethodCall(MethodCall call, ValueInstance? instance,
		ExecutionContext ctx)
	{
		var args = new List<ValueInstance>(call.Arguments.Count);
		foreach (var a in call.Arguments)
			args.Add(executor.RunExpression(a, ctx));
		if (instance is { ReturnType.IsDictionary: true } && args.Count > 0 && call.Method.Name == "Add")
		{
			if (args.Count == 2)
				((IDictionary)instance.Value!)[args[0]] = args[1];
			return instance;
		}
		var result = executor.Execute(call.Method, instance, args, ctx);
		if (call.Method.ReturnType.IsMutable && call.Instance is VariableCall variableCall &&
			instance != null)
			ctx.Set(variableCall.Variable.Name, new ValueInstance(instance.ReturnType, result.Value));
		return result;
	}

	private static ValueInstance Error(string name, ExecutionContext ctx, Expression? source = null)
	{
		var stacktraceList = new List<object?> { CreateStacktrace(ctx, source) };
		var errorMembers = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
		var errorType = ctx.Method.GetType(Base.Error);
		foreach (var member in errorType.Members)
			errorMembers[member.Name] = member.Type.Name switch
			{
				Base.Name or Base.Text => name,
				_ when member.Type.Name == Base.List || member.Type is GenericTypeImplementation
				{
					Generic.Name: Base.List
				} => stacktraceList,
				_ => throw new NotSupportedException("Error member not supported: " + member) //ncrunch: no coverage
			};
		return new ValueInstance(errorType, errorMembers);
	}

	private static Dictionary<string, object?> CreateStacktrace(ExecutionContext ctx,
		Expression? source)
	{
		var members = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
		var stacktraceType = ctx.Method.GetType(Base.Stacktrace);
		foreach (var member in stacktraceType.Members)
			members[member.Name] = member.Type.Name switch
			{
				Base.Method => CreateMethodValue(ctx.Method),
				Base.Text or Base.Name => ctx.Method.Type.FilePath,
				Base.Number => (double)(source?.LineNumber ?? ctx.Method.TypeLineNumber),
				_ => throw new NotSupportedException("Stacktrace member not supported: " + member) //ncrunch: no coverage
			};
		return members;
	}

	private static Dictionary<string, object?> CreateMethodValue(Method method)
	{
		var values = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
		var methodType = method.GetType(Base.Method);
		foreach (var member in methodType.Members)
			values[member.Name] = member.Type.Name switch
			{
				Base.Name or Base.Text => method.Name,
				Base.Type => CreateTypeValue(method.Type),
				_ => throw new NotSupportedException("Method member not supported: " + member) //ncrunch: no coverage
			};
		return values;
	}

	private static Dictionary<string, object?> CreateTypeValue(Type type)
	{
		var values = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
		var typeType = type.GetType(Base.Type);
		foreach (var member in typeType.Members)
			values[member.Name] = member.Type.Name switch
			{
				Base.Name => type.Name,
				Base.Text => type.Package.FullName,
				_ => throw new NotSupportedException("Type member not supported: " + member) //ncrunch: no coverage
			};
		return values;
	}
}