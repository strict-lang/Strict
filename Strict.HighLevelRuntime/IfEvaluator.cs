using Strict.Expressions;
using Strict.Language;

namespace Strict.HighLevelRuntime;

internal sealed class IfEvaluator(Executor executor)
{
	public ValueInstance Evaluate(If iff, ExecutionContext ctx) =>
		Executor.ToBool(executor.RunExpression(iff.Condition, ctx))
			? executor.RunExpression(iff.Then, ctx)
			: iff.OptionalElse != null
				? executor.RunExpression(iff.OptionalElse, ctx)
				: new ValueInstance(iff.ReturnType.GetType(Base.None), null);
}
