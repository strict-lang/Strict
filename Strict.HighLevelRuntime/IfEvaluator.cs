using Strict.Expressions;

namespace Strict.HighLevelRuntime;

internal sealed class IfEvaluator(Executor executor)
{
	public ValueInstance Evaluate(If iff, ExecutionContext ctx) =>
		Executor.ToBool(executor.RunExpression(iff.Condition, ctx))
			? executor.RunExpression(iff.Then, ctx)
			: iff.OptionalElse != null
				? executor.RunExpression(iff.OptionalElse, ctx)
				: executor.None();
}
