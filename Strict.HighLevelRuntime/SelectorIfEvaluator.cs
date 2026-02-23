using Strict.Expressions;
using Strict.Language;

namespace Strict.HighLevelRuntime;

internal sealed class SelectorIfEvaluator(Executor executor)
{
	public ValueInstance Evaluate(SelectorIf selectorIf, ExecutionContext ctx)
	{
		foreach (var @case in selectorIf.Cases)
			if (Executor.ToBool(executor.RunExpression(@case.Condition, ctx)))
				return executor.RunExpression(@case.Then, ctx);
		return selectorIf.OptionalElse != null
			? executor.RunExpression(selectorIf.OptionalElse, ctx)
			: new ValueInstance(selectorIf.ReturnType.GetType(Base.None), null);
	}
}
