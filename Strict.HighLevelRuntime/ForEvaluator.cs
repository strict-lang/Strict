using Strict.Expressions;
using Strict.Language;
using System.Text;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

internal sealed class ForEvaluator(Interpreter interpreter)
{
	public ValueInstance Evaluate(For f, ExecutionContext ctx)
	{
		interpreter.Statistics.ForCount++;
		var iterator = interpreter.RunExpression(f.Iterator, ctx);
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

	private ValueInstance TryEvaluate(For f, ExecutionContext ctx, ValueInstance iterator, ExecutionContext loop)
	{
		List<ValueInstance>? results = null;
		var itemType = GetForValueType(iterator);
		var iteratorInstance = iterator.TryGetValueTypeInstance();
		var isRangeIterator = iteratorInstance?.ReturnType == interpreter.rangeType;
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
			interpreter.listType.GetGenericImplementation(results is { Count: > 0 }
				? results[0].GetType()
				: f.Body.ReturnType), results?.ToArray() ?? []);
	}

	private void ExecuteForIteration(For f, ExecutionContext ctx, ValueInstance iterator,
		ref List<ValueInstance>? results, Type itemType, int index, ExecutionContext loop,
		bool isRangeIterator, Body? bodyAsBody)
	{
		var indexInstance = new ValueInstance(interpreter.numberType, index);
		loop.Variables[Type.IndexLowercase] = indexInstance;
		var value = iterator.IsPrimitiveType(interpreter.numberType) || isRangeIterator
			? indexInstance
			: iterator.GetIteratorValue(itemType, index);
		loop.Variables[Type.ValueLowercase] = value;
		foreach (var customVariable in f.CustomVariables)
			if (customVariable is VariableCall variableCall)
				loop.Variables[variableCall.Variable.Name] = value;
		var itemResult = bodyAsBody != null
			? EvaluateBody(bodyAsBody, loop)
			: interpreter.RunExpression(f.Body, loop);
		if (loop.ExitMethodAndReturnValue.HasValue)
			ctx.ExitMethodAndReturnValue = loop.ExitMethodAndReturnValue;
		else if (!itemResult.IsPrimitiveType(interpreter.noneType) && !itemResult.IsMutable)
		{
			results ??= new List<ValueInstance>();
			results.Add(itemResult);
		}
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
		{
			var sum = 0.0;
			if (results != null)
				for (var index = 0; index < results.Count; index++)
					sum += results[index].Number;
			return new ValueInstance(interpreter.numberType, sum);
		}
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
				text.Append((char)value.Number);
			else if (value.IsText)
				text.Append(value.Text);
			else if (value.IsPrimitiveType(interpreter.numberType))
				text.Append(value.GetCachedNumberString());
			else if (value.IsPrimitiveType(interpreter.booleanType))
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
        throw new InterpreterExecutionFailed(ctx.Method,
					InterpreterExecutionFailed.BuildContextMessage(ctx.Method, ctx.Method.TypeLineNumber,
						ctx, "For text return type cannot consolidate value " + value));
		return new ValueInstance(text.ToString());
	}

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
		return interpreter.numberType;
	}
}