using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

internal sealed class ForEvaluator(Executor executor)
{
	public ValueInstance Evaluate(For f, ExecutionContext ctx)
	{
		executor.Statistics.ForCount++;
		var iterator = executor.RunExpression(f.Iterator, ctx);
		List<ValueInstance>? results = null;
		var itemType = GetForValueType(iterator);
		var iteratorInstance = iterator.TryGetValueTypeInstance();
		var loop = new ExecutionContext(ctx.Type, ctx.Method) { This = ctx.This, Parent = ctx };
		if (iteratorInstance?.ReturnType == executor.rangeType &&
			iteratorInstance.Members.TryGetValue("Start", out var startValue) &&
			iteratorInstance.Members.TryGetValue("ExclusiveEnd", out var endValue))
		{
			var start = (int)startValue.Number;
			var end = (int)endValue.Number;
			if (start <= end)
				for (var index = start; index < end; index++)
				{
					loop.ResetForLoopIteration();
					ExecuteForIteration(f, ctx, iterator, ref results, itemType, index, loop);
					if (ctx.ExitMethodAndReturnValue.HasValue)
						return ctx.ExitMethodAndReturnValue.Value;
				}
			else
				for (var index = start; index > end; index--)
				{
					loop.ResetForLoopIteration();
					ExecuteForIteration(f, ctx, iterator, ref results, itemType, index, loop);
					if (ctx.ExitMethodAndReturnValue.HasValue)
						return ctx.ExitMethodAndReturnValue.Value;
				}
		}
		else
		{
			var loopRange = new Range(0, iterator.GetIteratorLength());
			for (var index = loopRange.Start.Value; index < loopRange.End.Value; index++)
			{
				loop.ResetForLoopIteration();
				ExecuteForIteration(f, ctx, iterator, ref results, itemType, index, loop);
				if (ctx.ExitMethodAndReturnValue.HasValue)
					return ctx.ExitMethodAndReturnValue.Value;
			}
		}
		return ShouldConsolidateForResult(results, ctx) ?? new ValueInstance(
			executor.listType.GetGenericImplementation(itemType), results ?? []);
	}

	private void ExecuteForIteration(For f, ExecutionContext ctx, ValueInstance iterator,
		ref List<ValueInstance>? results, Type itemType, int index, ExecutionContext loop)
	{
		var indexInstance = new ValueInstance(executor.numberType, index);
		loop.Set(Type.IndexLowercase, indexInstance);
		var value = iterator.IsPrimitiveType(executor.numberType) ||
			iterator.TryGetValueTypeInstance()?.ReturnType == executor.rangeType
				? indexInstance
				: iterator.GetIteratorValue(itemType, index);
		loop.Set(Type.ValueLowercase, value);
		foreach (var customVariable in f.CustomVariables)
			if (customVariable is VariableCall variableCall)
				loop.Set(variableCall.Variable.Name, value);
		var itemResult = f.Body is Body body
			? EvaluateBody(body, loop)
			: executor.RunExpression(f.Body, loop);
		if (loop.ExitMethodAndReturnValue.HasValue)
			ctx.ExitMethodAndReturnValue = loop.ExitMethodAndReturnValue;
		else if (!itemResult.IsPrimitiveType(executor.noneType) && !itemResult.IsMutable)
		{
			results ??= new List<ValueInstance>();
			results.Add(itemResult);
		}
	}

	private ValueInstance EvaluateBody(Body body, ExecutionContext ctx)
	{
		var last = executor.noneInstance;
		foreach (var e in body.Expressions)
		{
			last = executor.RunExpression(e, ctx);
			if (ctx.ExitMethodAndReturnValue.HasValue)
				return ctx.ExitMethodAndReturnValue.Value;
		}
		return last;
	}

	private ValueInstance? ShouldConsolidateForResult(List<ValueInstance>? results,
		ExecutionContext ctx)
	{
		if (ctx.Method.ReturnType.IsNumber)
			return new ValueInstance(executor.numberType,
				results?.Sum(value => value.Number) ?? 0);
		if (ctx.Method.ReturnType.IsBoolean)
			return new ValueInstance(executor.booleanType,
				results?.Any(value => value.Boolean) ?? false);
		if (!ctx.Method.ReturnType.IsText)
			return null;
		if (results == null)
			return new ValueInstance("");
		var text = "";
		foreach (var value in results)
			if (value.IsPrimitiveType(executor.characterType))
				text += (char)value.Number;
			else if (value.IsText || value.IsPrimitiveType(executor.numberType) ||
				value.IsPrimitiveType(executor.booleanType))
				text += value.ToExpressionCodeString();
			else if (value.IsList || value.IsDictionary)
				//TODO: need test
				text += (text == ""
					? ""
					: ", ") + "(" + value.ToExpressionCodeString() + ")";
			else
				throw new NotSupportedException("For text return type cannot consolidate value " + value);
		return new ValueInstance(text);
	}

	private Type GetForValueType(ValueInstance iterator) =>
		iterator.IsText
			? executor.characterType
			: iterator.IsList
				? iterator.GetIteratorType()
				: executor.numberType;
}