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
		var results = new List<ValueInstance>();
		var itemType = GetForValueType(iterator);
		if (iterator.ReturnType.Name == Base.Range &&
			iterator.Value is IDictionary<string, object?> rangeValues &&
			rangeValues.TryGetValue("Start", out var startValue) &&
			rangeValues.TryGetValue("ExclusiveEnd", out var endValue))
		{
			var start = Convert.ToInt32(startValue);
			var end = Convert.ToInt32(endValue);
			if (start <= end)
				for (var index = start; index < end; index++)
				{
					ExecuteForIteration(f, ctx, iterator, results, itemType, index);
					if (ctx.ExitMethodAndReturnValue.HasValue)
						return ctx.ExitMethodAndReturnValue.Value;
				}
			else
				for (var index = start; index > end; index--)
				{
					ExecuteForIteration(f, ctx, iterator, results, itemType, index);
					if (ctx.ExitMethodAndReturnValue.HasValue)
						return ctx.ExitMethodAndReturnValue.Value;
				}
		}
		else
		{
			var loopRange = new Range(0, iterator.GetIteratorLength());
			for (var index = loopRange.Start.Value; index < loopRange.End.Value; index++)
			{
				ExecuteForIteration(f, ctx, iterator, results, itemType, index);
				if (ctx.ExitMethodAndReturnValue.HasValue)
					return ctx.ExitMethodAndReturnValue.Value;
			}
		}
		return ShouldConsolidateForResult(results, ctx) ?? executor.CreateValueInstance(
			results.Count == 0
				? iterator.ReturnType
				: iterator.ReturnType.GetType(Base.List).GetGenericImplementation(results[0].ReturnType),
			results);
	}

	private void ExecuteForIteration(For f, ExecutionContext ctx, ValueInstance iterator,
		ICollection<ValueInstance> results, Type itemType, int index)
	{
		var loop = new ExecutionContext(ctx.Type, ctx.Method) { This = ctx.This, Parent = ctx };
		var indexInstance = executor.Number(itemType, index);
		loop.Set(Type.IndexLowercase, indexInstance);
		//If this is Range or Number, we should not call GetIteratorValue, index is what we use!
		var value = iterator.IsNumber || iterator.IsRange ? indexInstance : iterator.GetIteratorValue(itemType, index);
		if (itemType.IsText && value is char character)
			value = character.ToString();
		var valueInstance = value is ValueInstance vi
			? vi
			: executor.CreateValueInstance(itemType, value);
		loop.Set(Type.ValueLowercase, valueInstance);
		foreach (var customVariable in f.CustomVariables)
			if (customVariable is VariableCall variableCall)
				loop.Set(variableCall.Variable.Name, valueInstance);
		var itemResult = f.Body is Body body
			? EvaluateBody(body, loop)
			: executor.RunExpression(f.Body, loop);
		if (loop.ExitMethodAndReturnValue.HasValue)
			ctx.ExitMethodAndReturnValue = loop.ExitMethodAndReturnValue;
		else if (itemResult.IsPrimitiveType(executor.noneType) && !itemResult.IsMutable)
			results.Add(itemResult);
	}

	private ValueInstance EvaluateBody(Body body, ExecutionContext ctx)
	{
		var last = executor.None(body.Method);
		foreach (var e in body.Expressions)
		{
			last = executor.RunExpression(e, ctx);
			if (ctx.ExitMethodAndReturnValue.HasValue)
				return ctx.ExitMethodAndReturnValue.Value;
		}
		return last;
	}

	private ValueInstance? ShouldConsolidateForResult(List<ValueInstance> results,
		ExecutionContext ctx)
	{
		if (ctx.Method.ReturnType.IsNumber)
     return executor.Number(ctx.Method, results.Sum(value => value.Number()));
		if (ctx.Method.ReturnType.IsText)
		{
			var text = "";
			foreach (var value in results)
				text += value.ReturnType.Name switch
				{
					Base.Number => (int)value.Number(),
          Base.Character => "" + (char)(int)value.Value!,
					Base.Text => (string)value.Value!,
					_ => throw new NotSupportedException("Can't append to text: " + value)
				};
			return executor.CreateValueInstance(ctx.Method.ReturnType, text);
		}
		return ctx.Method.ReturnType.IsBoolean
			? executor.Bool(ctx.Method, results.Any(value => value.Boolean()))
			: null;
	}

	private static Type GetForValueType(ValueInstance iterator) =>
		iterator.ReturnType is GenericTypeImplementation { Generic.Name: Base.List } list
			? list.ImplementationTypes[0]
			: iterator.ReturnType.IsText
				? iterator.ReturnType.GetType(Base.Text)
				: iterator.ReturnType.GetType(Base.Number);
}