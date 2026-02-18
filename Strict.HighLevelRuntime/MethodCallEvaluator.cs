using Strict.Expressions;
using Strict.Language;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

internal sealed class MethodCallEvaluator
{
	private readonly Executor executor;
	private readonly Type boolType;
	private readonly Type numberType;
	private readonly Type genericListType;
	private readonly Type errorType;
	private readonly Type stacktraceType;
	private readonly Type methodType;
	private readonly Type typeType;

	public MethodCallEvaluator(Executor executor)
	{
		this.executor = executor;
		boolType = executor.BoolType;
		numberType = executor.NumberType;
		genericListType = executor.GenericListType;
		errorType = executor.ErrorType;
		stacktraceType = executor.StacktraceType;
		methodType = executor.MethodType;
		typeType = executor.TypeType;
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
		var args = new List<ValueInstance>(call.Arguments.Count);
		foreach (var a in call.Arguments)
			args.Add(executor.RunExpression(a, ctx));

   if (instance != null && instance.ReturnType is GenericTypeImplementation { Generic.Name: Base.Dictionary } &&
			instance.Value is IDictionary dictionaryValues && args.Count > 0 && call.Method.Name == "Add")
		{
      if (args.Count == 2)
			{
        var addKey = args[0].Value ??
					throw new InvalidOperationException("Dictionary key cannot be null");
				dictionaryValues[addKey] = args[1].Value;
				return instance;
			}
      if (TryGetPairValues(args[0], out var key, out var value))
			{
        var nonNullKey = key ??
					throw new InvalidOperationException("Dictionary key cannot be null");
				dictionaryValues[nonNullKey] = value;
				return instance;
			}
		}
		return executor.Execute(call.Method, instance, args, ctx);
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
			throw new InvalidOperationException(
				"Binary call must have instance and 1 argument");
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
				BinaryOperator.Plus => executor.Number(l + r),
				BinaryOperator.Minus => executor.Number(l - r),
				BinaryOperator.Multiply => executor.Number(l * r),
				BinaryOperator.Divide => executor.Number(l / r),
				BinaryOperator.Modulate => executor.Number(l % r),
				BinaryOperator.Power => executor.Number(Math.Pow(l, r)),
				_ => ExecuteMethodCall(call, leftInstance, ctx)
			};
		}
		if (leftInstance.ReturnType.Name == Base.Text && rightInstance.ReturnType.Name == Base.Text)
		{
			return op == BinaryOperator.Plus
				? new ValueInstance(leftInstance.ReturnType, (string)left! + (string)right!)
				: throw new NotSupportedException("Only + operator is supported for Text, got: " + op);
		}
		if (leftInstance.ReturnType.IsIterator && rightInstance.ReturnType.IsIterator)
		{
			if (left is not IList<ValueInstance> leftList ||
				right is not IList<ValueInstance> rightList)
				throw new InvalidOperationException(
					"Expected List<ValueInstance> for iterator operation, " +
					"other iterators are not yet supported: left=" + left + ", right=" + right);
			if (op is BinaryOperator.Multiply or BinaryOperator.Divide &&
				leftList.Count != rightList.Count)
				return Error(Executor.ListsHaveDifferentDimensions, ctx, call);
			return op switch
			{
				BinaryOperator.Plus => CombineLists(leftInstance.ReturnType, leftList, rightList),
				BinaryOperator.Minus => SubtractLists(leftInstance.ReturnType, leftList, rightList),
				BinaryOperator.Multiply => MultiplyLists(leftInstance.ReturnType, leftList, rightList),
				BinaryOperator.Divide => DivideLists(leftInstance.ReturnType, leftList, rightList),
				_ => throw new NotSupportedException(
					"Only +, -, *, / operators are supported for Lists, got: " + op)
			};
		}
		if (leftInstance.ReturnType.IsIterator && rightInstance.ReturnType.Name == Base.Number)
		{
			if (left is not IList<ValueInstance> leftList)
				throw new InvalidOperationException("Expected left list for iterator operation " + op +
					": left=" + left + ", right=" + right);
			if (op == BinaryOperator.Plus)
				return AddToList(leftInstance.ReturnType, leftList, rightInstance);
			if (op == BinaryOperator.Minus)
				return RemoveFromList(leftInstance.ReturnType, leftList, rightInstance);
			if (right is not double rightNumber)
				throw new InvalidOperationException("Expected right number for iterator operation " + op +
					": left=" + left + ", right=" + right);
			if (op == BinaryOperator.Multiply)
				return MultiplyList(leftInstance.ReturnType, leftList, rightNumber);
			if (op == BinaryOperator.Divide)
				return DivideList(leftInstance.ReturnType, leftList, rightNumber);
			throw new NotSupportedException(
				"Only +, -, *, / operators are supported for List and Number, got: " + op);
		}
		return ExecuteMethodCall(call, leftInstance, ctx);
	}

	private ValueInstance ExecuteComparisonOperation(MethodCall call, ExecutionContext ctx,
		ValueInstance leftInstance, ValueInstance rightInstance)
	{
		var op = call.Method.Name;
		var left = leftInstance.Value;
		var right = rightInstance.Value;
		if (op is BinaryOperator.Is or UnaryOperator.Not)
		{
			if (left == null || right == null)
				throw new Executor.ComparisonsToNullAreNotAllowed(call.Method, left, right);
			if (rightInstance.ReturnType.IsError)
			{
				var matches = rightInstance.ReturnType.Name == Base.Error
					? leftInstance.ReturnType.IsError
					: leftInstance.ReturnType.IsSameOrCanBeUsedAs(rightInstance.ReturnType);
				return op is BinaryOperator.Is
					? executor.Bool(matches)
					: executor.Bool(!matches);
			}
			if (call.Instance!.ReturnType.Name == Base.Character && right is string rightText)
			{
				right = (int)rightText[0];
				rightInstance = new ValueInstance(call.Instance.ReturnType, right);
			}
			if (call.Instance.ReturnType.Name == Base.Text && right is int rightInt)
			{
				right = rightInt + "";
				rightInstance = new ValueInstance(call.Instance.ReturnType, right);
			}
			var equals = leftInstance.Equals(rightInstance);
			return op is BinaryOperator.Is
				? executor.Bool(equals)
				: executor.Bool(!equals);
		}
		var l = EqualsExtensions.NumberToDouble(left);
		var r = EqualsExtensions.NumberToDouble(right);
		return op switch
		{
			BinaryOperator.Greater => executor.Bool(l > r),
			BinaryOperator.Smaller => executor.Bool(l < r),
			BinaryOperator.GreaterOrEqual => executor.Bool(l >= r),
			BinaryOperator.SmallerOrEqual => executor.Bool(l <= r),
			_ => ExecuteMethodCall(call, leftInstance, ctx)
		};
	}

	private ValueInstance ExecuteBinaryOperation(MethodCall call, ExecutionContext ctx,
		ValueInstance leftInstance, ValueInstance rightInstance)
	{
		var left = leftInstance.Value;
		var right = rightInstance.Value;
		return call.Method.Name switch
		{
			BinaryOperator.And => executor.Bool(Executor.ToBool(left) && Executor.ToBool(right)),
			BinaryOperator.Or => executor.Bool(Executor.ToBool(left) || Executor.ToBool(right)),
			BinaryOperator.Xor => executor.Bool(Executor.ToBool(left) ^ Executor.ToBool(right)),
			_ => ExecuteMethodCall(call, leftInstance, ctx)
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
			string text => text,
			double number => number.ToString(CultureInfo.InvariantCulture),
			int number => number.ToString(CultureInfo.InvariantCulture),
			_ => value?.ToString() ?? string.Empty
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

	private ValueInstance ExecuteMethodCall(MethodCall call, ValueInstance instance,
		ExecutionContext ctx)
	{
		var args = new List<ValueInstance>(call.Arguments.Count);
		foreach (var a in call.Arguments)
			args.Add(executor.RunExpression(a, ctx));
		return executor.Execute(call.Method, instance, args, ctx);
	}

	private ValueInstance Error(string name, ExecutionContext ctx, Expression? source = null)
	{
		var stacktraceList = new List<object?> { CreateStacktrace(ctx, source) };
		var errorMembers = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
		foreach (var member in errorType.Members)
			errorMembers[member.Name] = member.Type.Name switch
			{
				Base.Name or Base.Text => name,
				_ when member.Type.Name == Base.List || member.Type is GenericTypeImplementation
				{
					Generic.Name: Base.List
				} => stacktraceList,
				_ => throw new NotSupportedException("Error member not supported: " + member)
			};
		return new ValueInstance(errorType, errorMembers);
	}

	private Dictionary<string, object?> CreateStacktrace(ExecutionContext ctx, Expression? source)
	{
		var members = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
		foreach (var member in stacktraceType.Members)
			members[member.Name] = member.Type.Name switch
			{
				Base.Method => CreateMethodValue(ctx.Method),
				Base.Text or Base.Name => ctx.Method.Type.FilePath,
				Base.Number => (double)(source?.LineNumber ?? ctx.Method.TypeLineNumber),
				_ => throw new NotSupportedException("Stacktrace member not supported: " + member)
			};
		return members;
	}

	private Dictionary<string, object?> CreateMethodValue(Method method)
	{
		var values = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
		foreach (var member in methodType.Members)
			values[member.Name] = member.Type.Name switch
			{
				Base.Name or Base.Text => method.Name,
				Base.Type => CreateTypeValue(method.Type),
				_ => throw new NotSupportedException("Method member not supported: " + member)
			};
		return values;
	}

	private Dictionary<string, object?> CreateTypeValue(Type type)
	{
		var values = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
		foreach (var member in typeType.Members)
			values[member.Name] = member.Type.Name switch
			{
				Base.Name => type.Name,
				Base.Text => type.Package.FullName,
				_ => throw new NotSupportedException("Type member not supported: " + member)
			};
		return values;
	}

	private static bool TryGetPairValues(ValueInstance pair, out object? key, out object? value)
	{
		key = null;
		value = null;
		var pairList = pair.Value switch
		{
			IList list => list,
			_ => null
		};
		if (pairList == null || pairList.Count < 2)
			return false;
		key = pairList[0] is ValueInstance keyInstance ? keyInstance.Value : pairList[0];
		value = pairList[1] is ValueInstance valueInstance ? valueInstance.Value : pairList[1];
		return true;
	}
}
