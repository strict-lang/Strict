using Strict.Expressions;

namespace Strict.HighLevelRuntime;

internal sealed class SelectorIfEvaluator(Interpreter interpreter)
{
	public ValueInstance Evaluate(SelectorIf selectorIf, ExecutionContext ctx)
	{
		interpreter.Statistics.SelectorIfCount++;
		foreach (var @case in selectorIf.Cases)
			if (interpreter.RunExpression(@case.Condition, ctx).Boolean)
				return interpreter.RunExpression(@case.Then, ctx);
		return selectorIf.OptionalElse != null
			? interpreter.RunExpression(selectorIf.OptionalElse, ctx)
			: interpreter.noneInstance;
	}
}