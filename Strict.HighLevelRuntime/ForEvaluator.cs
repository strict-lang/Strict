using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

internal sealed class ForEvaluator(Executor executor)
{
	public ValueInstance Evaluate(For f, ExecutionContext ctx)
	{
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
					ExecuteForIteration(f, ctx, iterator, results, itemType, index);
			else
				for (var index = start; index > end; index--)
					ExecuteForIteration(f, ctx, iterator, results, itemType, index);
		}
		else
		{
			var loopRange = new Range(0, iterator.GetIteratorLength());
			for (var index = loopRange.Start.Value; index < loopRange.End.Value; index++)
				ExecuteForIteration(f, ctx, iterator, results, itemType, index);
		}
		return ShouldConsolidateForResult(results, ctx) ?? new ValueInstance(results.Count == 0
				? iterator.ReturnType
				: iterator.ReturnType.GetType(Base.List).GetGenericImplementation(results[0].ReturnType),
			results);
	}

	private void ExecuteForIteration(For f, ExecutionContext ctx, ValueInstance iterator,
		ICollection<ValueInstance> results, Type itemType, int index)
	{
		var loop = new ExecutionContext(ctx.Type, ctx.Method) { This = ctx.This, Parent = ctx };
		loop.Set(Type.IndexLowercase, new ValueInstance(itemType.GetType(Base.Number), index));
		var value = iterator.GetIteratorValue(index);
		var valueInstance = value as ValueInstance ?? new ValueInstance(itemType, value);
		loop.Set(Type.ValueLowercase, valueInstance);
		foreach (var customVariable in f.CustomVariables)
			if (customVariable is VariableCall variableCall)
				loop.Set(variableCall.Variable.Name, valueInstance);
    var itemResult = f.Body is Body body
			? EvaluateBody(body, loop)
			: executor.RunExpression(f.Body, loop);
		if (itemResult.ReturnType.Name != Base.None)
			results.Add(itemResult);
	}

	private ValueInstance EvaluateBody(Body body, ExecutionContext ctx)
	{
		var noneType = (ctx.This?.ReturnType.Package ?? body.Method.Type.Package).FindType(Base.None)!;
		ValueInstance last = new(noneType, null);
		foreach (var e in body.Expressions)
			last = executor.RunExpression(e, ctx);
		return last;
	}

	private static ValueInstance? ShouldConsolidateForResult(List<ValueInstance> results,
		ExecutionContext ctx)
	{
		if (ctx.Method.ReturnType.Name == Base.Number)
			return new ValueInstance(ctx.Method.ReturnType,
				results.Sum(value => EqualsExtensions.NumberToDouble(value.Value)));
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
			return new ValueInstance(ctx.Method.ReturnType, text);
		}
		return ctx.Method.ReturnType.Name == Base.Boolean
			? new ValueInstance(ctx.Method.ReturnType, results.Any(value => value.Value is true))
			: null;
	}

	private static Type GetForValueType(ValueInstance iterator) =>
		iterator.ReturnType is GenericTypeImplementation { Generic.Name: Base.List } list
			? list.ImplementationTypes[0]
			: iterator.ReturnType.Name == Base.Text
       ? iterator.ReturnType.GetType(Base.Text)
				: iterator.ReturnType.GetType(Base.Number);
}