using Strict.Expressions;
using Strict.Language;
using System;
using System.Collections;
using System.Collections.Generic;

namespace Strict.HighLevelRuntime;

internal sealed class ListCallEvaluator(Executor executor)
{
	public ValueInstance Evaluate(ListCall call, ExecutionContext ctx)
	{
		var listInstance = executor.RunExpression(call.List, ctx);
		var indexValue = executor.RunExpression(call.Index, ctx);
		var index = Convert.ToInt32(EqualsExtensions.NumberToDouble(indexValue.Value));
		if (listInstance.Value is IList list)
			return list[index] as ValueInstance ?? new ValueInstance(call.ReturnType, list[index]);
		if (listInstance.Value is IDictionary<string, object?> members &&
			(members.TryGetValue("Elements", out var elements) ||
				members.TryGetValue("elements", out elements)) && elements is IList memberList)
			return memberList[index] as ValueInstance ??
				new ValueInstance(call.ReturnType, memberList[index]);
		if (listInstance.Value is string text)
			return new ValueInstance(call.ReturnType, (int)text[index]);
		throw new InvalidOperationException("List call can only be used on iterators, got: " +
			listInstance);
	}
}
