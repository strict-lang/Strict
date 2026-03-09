using Strict.Expressions;
using Strict.Language;
using System.Text;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

internal sealed class ForEvaluator(Executor executor)
{
	public ValueInstance Evaluate(For f, ExecutionContext ctx)
	{
		executor.Statistics.ForCount++;
		var iterator = executor.RunExpression(f.Iterator, ctx);
		var loop = executor.RentContext(ctx.Type, ctx.Method, ctx.This, ctx);
		try
		{
			return TryEvaluate(f, ctx, iterator, loop);
		}
		finally
		{
			executor.ReturnContext(loop);
		}
	}

	private ValueInstance TryEvaluate(For f, ExecutionContext ctx, ValueInstance iterator, ExecutionContext loop)
	{
		List<ValueInstance>? results = null;
		var itemType = GetForValueType(iterator);
		var iteratorInstance = iterator.TryGetValueTypeInstance();
		var isRangeIterator = iteratorInstance?.ReturnType == executor.rangeType;
		var bodyAsBody = f.Body as Body;
		if (isRangeIterator &&
			iteratorInstance!.TryGetValue("Start", out var startValue) &&
			iteratorInstance.TryGetValue("ExclusiveEnd", out var endValue))
		{
			var start = (int)startValue.Number;
			var end = (int)endValue.Number;
			if (start <= end)
				for (var index = start; index < end; index++)
				{
					loop.ResetIteration();
					ExecuteForIteration(f, ctx, iterator, ref results, itemType, index, loop, isRangeIterator, bodyAsBody);
					if (ctx.ExitMethodAndReturnValue.HasValue)
						return ctx.ExitMethodAndReturnValue.Value;
				}
			else
				for (var index = start; index > end; index--)
				{
					loop.ResetIteration();
					ExecuteForIteration(f, ctx, iterator, ref results, itemType, index, loop, isRangeIterator, bodyAsBody);
					if (ctx.ExitMethodAndReturnValue.HasValue)
						return ctx.ExitMethodAndReturnValue.Value;
				}
		}
		else
		{
			var loopRange = new Range(0, iterator.GetIteratorLength());
			for (var index = loopRange.Start.Value; index < loopRange.End.Value; index++)
			{
				loop.ResetIteration();
				ExecuteForIteration(f, ctx, iterator, ref results, itemType, index, loop, isRangeIterator, bodyAsBody);
				if (ctx.ExitMethodAndReturnValue.HasValue)
					return ctx.ExitMethodAndReturnValue.Value;
			}
		}
		return ShouldConsolidateForResult(results, ctx) ?? new ValueInstance(
			executor.listType.GetGenericImplementation(itemType), results?.ToArray() ?? []);
	}

	private void ExecuteForIteration(For f, ExecutionContext ctx, ValueInstance iterator,
		ref List<ValueInstance>? results, Type itemType, int index, ExecutionContext loop,
		bool isRangeIterator, Body? bodyAsBody)
	{
		var indexInstance = new ValueInstance(executor.numberType, index);
		loop.Set(Type.IndexLowercase, indexInstance);
		var value = iterator.IsPrimitiveType(executor.numberType) || isRangeIterator
			? indexInstance
			: iterator.GetIteratorValue(itemType, index);
		loop.Set(Type.ValueLowercase, value);
		foreach (var customVariable in f.CustomVariables)
			if (customVariable is VariableCall variableCall)
				loop.Set(variableCall.Variable.Name, value);
		var itemResult = bodyAsBody != null
			? EvaluateBody(bodyAsBody, loop)
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
		{
			var sum = 0.0;
			if (results != null)
				for (var i = 0; i < results.Count; i++)
					sum += results[i].Number;
			return new ValueInstance(executor.numberType, sum);
		}
		if (ctx.Method.ReturnType.IsBoolean)
		{
			var any = false;
			if (results != null)
				for (var i = 0; i < results.Count; i++)
					if (results[i].Boolean)
					{
						any = true;
						break;
					}
			return new ValueInstance(executor.booleanType, any);
		}
		if (!ctx.Method.ReturnType.IsText)
			return null;
		if (results == null)
			return new ValueInstance("");
		var text = new StringBuilder();
		foreach (var value in results)
			if (value.IsPrimitiveType(executor.characterType))
				text.Append((char)value.Number);
			else if (value.IsText)
				text.Append(value.Text);
			else if (value.IsPrimitiveType(executor.numberType))
				text.Append(value.GetCachedNumberString());
			else if (value.IsPrimitiveType(executor.booleanType))
				text.Append(value.Boolean //ncrunch: no coverage
					? "true"
					: "false");
			else if (value.IsList || value.IsDictionary)
			{
				if (text.Length > 0)
					text.Append(", ");
				text.Append('(');
				text.Append(value.ToExpressionCodeString());
				text.Append(')');
			}
			else
				throw new NotSupportedException("For text return type cannot consolidate value " + value);
		return new ValueInstance(text.ToString());
	}

	private Type GetForValueType(ValueInstance iterator) =>
		iterator.IsText
			? executor.characterType
			: iterator.IsList
				? iterator.GetIteratorType()
				: executor.numberType;
}