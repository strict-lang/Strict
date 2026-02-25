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
					if (ctx.ExitMethodAndReturnValue != null)
						return ctx.ExitMethodAndReturnValue;
				}
			else
				for (var index = start; index > end; index--)
				{
					ExecuteForIteration(f, ctx, iterator, results, itemType, index);
					if (ctx.ExitMethodAndReturnValue != null)
						return ctx.ExitMethodAndReturnValue;
				}
		}
		else
		{
			var loopRange = new Range(0, iterator.GetIteratorLength());
			for (var index = loopRange.Start.Value; index < loopRange.End.Value; index++)
			{
				ExecuteForIteration(f, ctx, iterator, results, itemType, index);
				if (ctx.ExitMethodAndReturnValue != null)
					return ctx.ExitMethodAndReturnValue;
			}
		}
		return ShouldConsolidateForResult(results, ctx) ?? new ValueInstance(results.Count == 0
				? iterator.ReturnType
				: iterator.ReturnType.GetType(Base.List).GetGenericImplementation(results[0].ReturnType),
			results, executor.Statistics);
	}

	private void ExecuteForIteration(For f, ExecutionContext ctx, ValueInstance iterator,
		ICollection<ValueInstance> results, Type itemType, int index)
	{
		var loop = new ExecutionContext(ctx.Type, ctx.Method) { This = ctx.This, Parent = ctx };
		loop.Set(Type.IndexLowercase, new ValueInstance(itemType.GetType(Base.Number), index, executor.Statistics));
		var value = iterator.GetIteratorValue(index);
		if (itemType.Name == Base.Text && value is char character)
			value = character.ToString();
		var valueInstance = value as ValueInstance ?? new ValueInstance(itemType, value, executor.Statistics);
		loop.Set(Type.ValueLowercase, valueInstance);
		foreach (var customVariable in f.CustomVariables)
			if (customVariable is VariableCall variableCall)
				loop.Set(variableCall.Variable.Name, valueInstance);
		var itemResult = f.Body is Body body
			? EvaluateBody(body, loop)
			: executor.RunExpression(f.Body, loop);
		if (loop.ExitMethodAndReturnValue != null)
			ctx.ExitMethodAndReturnValue = loop.ExitMethodAndReturnValue;
		else if (itemResult.ReturnType.Name != Base.None && !itemResult.ReturnType.IsMutable)
			results.Add(itemResult);
	}

	private ValueInstance EvaluateBody(Body body, ExecutionContext ctx)
	{
		var noneType =
			(ctx.This?.ReturnType.Package ?? body.Method.Type.Package).FindType(Base.None)!;
		ValueInstance last = new(noneType, null, executor.Statistics);
		foreach (var e in body.Expressions)
		{
			last = executor.RunExpression(e, ctx);
			if (ctx.ExitMethodAndReturnValue != null)
				return ctx.ExitMethodAndReturnValue;
		}
		return last;
	}

	private ValueInstance? ShouldConsolidateForResult(List<ValueInstance> results,
		ExecutionContext ctx)
	{
		if (ctx.Method.ReturnType.Name == Base.Number)
			return new ValueInstance(ctx.Method.ReturnType,
				results.Sum(value => EqualsExtensions.NumberToDouble(value.Value)), executor.Statistics);
		if (ctx.Method.ReturnType.Name == Base.Text)
		{
			var text = "";
			foreach (var value in results)
				text += value.ReturnType.Name switch
				{
					Base.Number => (int)EqualsExtensions.NumberToDouble(value.Value),
					Base.Character => "" + (char)value.Value!,
					Base.Text => (string)value.Value!,
					_ => throw new NotSupportedException("Can't append to text: " + value)
				};
			return new ValueInstance(ctx.Method.ReturnType, text, executor.Statistics);
		}
		return ctx.Method.ReturnType.Name == Base.Boolean
			? new ValueInstance(ctx.Method.ReturnType, results.Any(value => value.Value is true), executor.Statistics)
			: null;
	}

	private static Type GetForValueType(ValueInstance iterator) =>
		iterator.ReturnType is GenericTypeImplementation { Generic.Name: Base.List } list
			? list.ImplementationTypes[0]
			: iterator.ReturnType.Name == Base.Text
				? iterator.ReturnType.GetType(Base.Text)
				: iterator.ReturnType.GetType(Base.Number);
}