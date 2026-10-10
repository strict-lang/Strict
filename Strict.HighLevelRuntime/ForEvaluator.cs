using System.Text;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

internal sealed class ForEvaluator(Interpreter interpreter)
{
	public ValueInstance Evaluate(For f, ExecutionContext ctx)
	{
		interpreter.Statistics.ForCount++;
		var iterator = MaterializeIteratorIfNeeded(
			Interpreter.CopyIfMutableList(interpreter.RunExpression(f.Iterator, ctx)));
		var loop = interpreter.RentContext(ctx.Type, ctx.Method, ctx.This, ctx);
		try
		{
			return TryEvaluate(f, ctx, iterator, loop);
		}
		finally
		{
			interpreter.ReturnContext(loop);
		}
	}

	private ValueInstance MaterializeIteratorIfNeeded(ValueInstance iterator)
	{
		if (iterator.IsText || iterator.IsList || iterator.IsPrimitiveType(interpreter.numberType))
			return iterator;
		var typeInstance = iterator.TryGetValueTypeInstance();
		if (typeInstance == null)
			return iterator;
		var iteratorMethod = typeInstance.ReturnType.Methods.FirstOrDefault(method =>
			method.Name == Keyword.For);
		if (iteratorMethod?.ReturnType.IsIterator != true)
			return iterator;
		var materialized = interpreter.Execute(iteratorMethod, iterator, []);
		return TryFlattenNestedIteratorList(materialized);
	}

	private ValueInstance TryFlattenNestedIteratorList(ValueInstance materialized)
	{
		if (!materialized.IsList || materialized.List.Items.Count == 0)
			return materialized;
		if (!materialized.List.Items.All(item => item.IsList))
			return materialized;
		var flattenedItems = new List<ValueInstance>();
		foreach (var nested in materialized.List.Items)
			flattenedItems.AddRange(nested.List.Items);
		if (flattenedItems.Count == 0)
			return materialized;
		var flattenedElementType = flattenedItems[0].GetType();
		return new ValueInstance(interpreter.listType.GetGenericImplementation(flattenedElementType),
			flattenedItems.ToArray());
	}

	private ValueInstance TryEvaluate(For f, ExecutionContext ctx, ValueInstance iterator,
		ExecutionContext loop)
	{
		List<ValueInstance>? results = null;
		var itemType = GetForValueType(iterator);
		var iteratorInstance = iterator.TryGetValueTypeInstance();
		var isRangeIterator = iteratorInstance?.ReturnType == interpreter.rangeType;
		var isNumberOnlyIteration = isRangeIterator || iterator.IsNumberLike(interpreter.numberType);
		var bodyAsBody = f.Body as Body;
		var start = 0;
		int end;
		if (isRangeIterator && iteratorInstance!.TryGetValue("Start", out var startValue) &&
			iteratorInstance.TryGetValue("ExclusiveEnd", out var endValue))
		{
			start = (int)startValue.GetArithmeticNumber();
			end = (int)endValue.GetArithmeticNumber();
		}
		else
		{
			end = iterator.GetIteratorLength();
			if (end < 0)
				throw new Interpreter.NegativeLoopCount(ctx.Method, end);
		}
		var step = start <= end
			? 1
			: -1;
		ValueInstance? outer = null;
		for (var index = start; index != end; index += step)
		{
			interpreter.ResetIteration(loop);
			loop.Variables[Type.OuterLowercase] =
				outer ??= ctx.Get(Type.ValueLowercase, interpreter.Statistics);
			ExecuteForIteration(f, ctx, iterator, ref results, itemType, index, loop,
				isNumberOnlyIteration, bodyAsBody);
			if (ctx.ExitMethodAndReturnValue.HasValue)
				return ctx.ExitMethodAndReturnValue.Value;
		}
		return ShouldConsolidateForResult(results, ctx) ?? new ValueInstance(
			interpreter.listType.GetGenericImplementation(results is { Count: > 0 }
				? GetResultElementType(results[0])
				: bodyAsBody?.Expressions[^1].ReturnType ?? f.Body.ReturnType), results?.ToArray() ?? []);
	}

	private void ExecuteForIteration(For f, ExecutionContext ctx, ValueInstance iterator,
		ref List<ValueInstance>? results, Type itemType, int index, ExecutionContext loop,
		bool isNumberOnlyIteration, Body? bodyAsBody)
	{
		var indexInstance = new ValueInstance(interpreter.numberType, index);
		loop.Variables[Type.IndexLowercase] = indexInstance;
		var iterationValue = isNumberOnlyIteration
			? indexInstance
			: iterator.GetIteratorValue(itemType, index);
		loop.Variables[Type.ValueLowercase] = iterationValue;
		AssignCustomVariables(f, ctx, loop, iterationValue);
		var itemResult = bodyAsBody != null
			? EvaluateBody(bodyAsBody, loop)
			: interpreter.RunExpression(f.Body, loop);
		if (loop.ExitMethodAndReturnValue.HasValue)
		{
			ctx.ExitMethodAndReturnValue = loop.ExitMethodAndReturnValue;
		}
		else if (!itemResult.IsPrimitiveType(interpreter.noneType) && (!itemResult.IsMutable ||
			IsVariableRead(bodyAsBody?.Expressions[^1] ?? f.Body)))
		{
			results ??= new List<ValueInstance>();
			if (results.Count == 1 && ctx.Method.ReturnType.IsNumber)
				results[0] = new ValueInstance(interpreter.numberType,
					Consolidate(results[0].GetArithmeticNumber(), itemResult.GetArithmeticNumber(),
						f.ShorthandOperator));
			else
				results.Add(Interpreter.CopyIfMutableList(itemResult));
		}
	}

	/// <summary>
	/// Changing lines like list.Add(value) give Mutable values that are not collected, a line only
	/// reading a changed variable collects its value of this iteration.
	/// </summary>
	private static bool IsVariableRead(Expression expression) =>
		expression is VariableCall or ParameterCall or MemberCall or Instance;

	private static void AssignCustomVariables(For f, ExecutionContext ctx, ExecutionContext loop,
		ValueInstance value)
	{
		if (f.CustomVariables.Length == 0)
			return;
		if (f.CustomVariables.Length == 1)
		{
			if (f.CustomVariables[0] is VariableCall variableCall)
				loop.Variables[variableCall.Variable.Name] = value;
			return;
		}
		var loopValues = GetLoopVariableValues(f, ctx, value);
		for (var index = 0; index < f.CustomVariables.Length; index++)
			if (f.CustomVariables[index] is VariableCall variableCall)
				loop.Variables[variableCall.Variable.Name] = loopValues[index];
	}

	private static IReadOnlyList<ValueInstance> GetLoopVariableValues(For f, ExecutionContext ctx,
		ValueInstance value)
	{
		if (value.IsList)
			return value.List.Items;
		var typeInstance = value.TryGetValueTypeInstance();
		if (typeInstance != null)
			for (var index = 0; index < typeInstance.Values.Length; index++)
				if (!typeInstance.ReturnType.Members[index].IsConstant && typeInstance.Values[index].IsList)
					return typeInstance.Values[index].List.Items;
		throw new InterpreterExecutionFailed(ctx.Method,
			InterpreterExecutionFailed.BuildContextMessage(ctx.Method, f.LineNumber, ctx,
				"Cannot split loop value " + value + " into " + f.CustomVariables.Length + " variables"));
	}

	private ValueInstance EvaluateBody(Body body, ExecutionContext ctx)
	{
		var last = interpreter.noneInstance;
		foreach (var e in body.Expressions)
		{
			last = interpreter.RunExpression(e, ctx);
			if (ctx.ExitMethodAndReturnValue.HasValue)
				return ctx.ExitMethodAndReturnValue.Value;
		}
		return last;
	}

	private ValueInstance? ShouldConsolidateForResult(List<ValueInstance>? results,
		ExecutionContext ctx)
	{
		if (ctx.Method.ReturnType.IsNumber)
			return new ValueInstance(interpreter.numberType, results is { Count: > 0 }
				? results[0].GetArithmeticNumber()
				: 0);
		if (ctx.Method.ReturnType.IsBoolean)
		{
			var any = false;
			if (results != null)
				for (var index = 0; index < results.Count; index++)
					if (results[index].Boolean)
					{
						any = true;
						break;
					}
			return new ValueInstance(interpreter.booleanType, any);
		}
		if (!ctx.Method.ReturnType.IsText)
			return null;
		if (results == null)
			return new ValueInstance("");
		var text = new StringBuilder();
		foreach (var value in results)
			if (value.IsPrimitiveType(interpreter.characterType))
			{
				text.Append((char)value.Number);
			}
			else if (value.IsText)
			{
				text.Append(value.Text);
			}
			else if (value.IsPrimitiveType(interpreter.numberType))
			{
				text.Append(value.GetCachedNumberString());
			}
			else if (value.IsPrimitiveType(interpreter.booleanType))
			{
				text.Append(value.Boolean //ncrunch: no coverage
					? "true"
					: "false");
			}
			else if (value.IsList || value.IsDictionary)
			{
				if (text.Length > 0)
					text.Append(", ");
				text.Append('(');
				text.Append(value.ToExpressionCodeString());
				text.Append(')');
			}
			else
			{
				throw new InterpreterExecutionFailed(ctx.Method,
					InterpreterExecutionFailed.BuildContextMessage(ctx.Method, ctx.Method.TypeLineNumber, ctx,
						"For text return type cannot consolidate value " + value));
			}
		return new ValueInstance(text.ToString());
	}

	/// <summary>
	/// Number loops fold each result into the first one right away, a Length loop over a long Text
	/// would otherwise keep one result per character.
	/// </summary>
	private static double Consolidate(double consolidated, double value,
		string shorthandOperator) =>
		shorthandOperator switch
		{
			BinaryOperator.Multiply => consolidated * value,
			BinaryOperator.Minus => consolidated - value,
			BinaryOperator.Divide => consolidated / value,
			BinaryOperator.Modulate => consolidated % value,
			BinaryOperator.Power => Math.Pow(consolidated, value),
			_ => consolidated + value
		};

	private Type GetResultElementType(ValueInstance result) =>
		result.IsText
			? interpreter.textType
			: result.IsList
				? result.List.ReturnType
				: result.TryGetValueTypeInstance()?.ReturnType ?? result.GetType();

	private Type GetForValueType(ValueInstance iterator)
	{
		if (iterator.IsText)
			return interpreter.characterType;
		if (iterator.IsList)
			return iterator.GetIteratorType();
		var typeInstance = iterator.TryGetValueTypeInstance();
		if (typeInstance?.ReturnType.IsList == true)
		{
			for (var index = 0; index < typeInstance.Values.Length; index++)
				if (typeInstance.Values[index].IsText)
					return interpreter.characterType;
			if (typeInstance.TryGetValue("elements", out var elementsMember) && elementsMember.IsList)
				return elementsMember.GetIteratorType();
		}
		var iteratorMethod = typeInstance?.ReturnType.Methods.FirstOrDefault(method =>
			method.Name == Keyword.For);
		return iteratorMethod?.ReturnType.IsIterator == true
			? iteratorMethod.ReturnType.GetFirstImplementation()
			: interpreter.numberType;
	}
}