using Strict.Expressions;
using Strict.Language;
using System;
using System.Collections.Generic;
using System.Linq;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

internal sealed class ForEvaluator
{
	private readonly Executor executor;
	private readonly Type numberType;
	private readonly Type genericListType;

	public ForEvaluator(Executor executor)
	{
		this.executor = executor;
		numberType = executor.NumberType;
		genericListType = executor.GenericListType;
	}

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
			var loopRange = iterator.ReturnType.Name == Base.Range
				? iterator.GetRange()
				: new Range(0, iterator.GetIteratorLength());
			for (var index = loopRange.Start.Value; index < loopRange.End.Value; index++)
				ExecuteForIteration(f, ctx, iterator, results, itemType, index);
		}
		return ShouldConsolidateForResult(results, ctx) ?? new ValueInstance(results.Count == 0
			? iterator.ReturnType
			: genericListType.GetGenericImplementation(results[0].ReturnType), results);
	}

	private void ExecuteForIteration(For f, ExecutionContext ctx, ValueInstance iterator,
		ICollection<ValueInstance> results, Type itemType, int index)
	{
		var loopContext =
			new ExecutionContext(ctx.Type, ctx.Method) { This = ctx.This, Parent = ctx };
		loopContext.Set(Type.IndexLowercase, new ValueInstance(numberType, index));
		var value = iterator.GetIteratorValue(index);
		loopContext.Set(Type.ValueLowercase,
			value as ValueInstance ?? new ValueInstance(itemType, value));
		foreach (var customVariable in f.CustomVariables)
			if (customVariable is VariableCall variableCall)
				loopContext.Set(variableCall.Variable.Name,
					new ValueInstance(variableCall.ReturnType, value));
		var itemResult = executor.RunExpression(f.Body, loopContext);
		if (itemResult.ReturnType.Name != Base.None)
			results.Add(itemResult);
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
					Base.Number or Base.Character => (int)EqualsExtensions.NumberToDouble(value.Value),
					Base.Text => ((string)value.Value!)[0],
					_ => throw new NotSupportedException("Can't append to text: " + value)
				};
			return new ValueInstance(ctx.Method.ReturnType, text);
		}
		if (ctx.Method.ReturnType.Name == Base.Boolean)
			return new ValueInstance(ctx.Method.ReturnType, results.Any(value => value.Value is true));
		return null;
	}

	private Type GetForValueType(ValueInstance iterator) =>
		iterator.ReturnType is GenericTypeImplementation { Generic.Name: Base.List } list
			? list.ImplementationTypes[0]
			: numberType;
}
