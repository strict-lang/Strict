using Strict.Expressions;
using Strict.Language;

namespace Strict.HighLevelRuntime;

internal sealed class IfEvaluator(Executor executor)
{
	public ValueInstance Evaluate(If iff, ExecutionContext ctx)
	{
		executor.Statistics.IfCount++;
		if (Executor.ToBool(executor.RunExpression(iff.Condition, ctx)))
		{
			var thenResult = executor.RunExpression(iff.Then, ctx);
			return iff.Then is MutableReassignment || IsMutableInstanceCall(iff.Then)
				? executor.None(iff.ReturnType)
				: thenResult;
		}
		if (iff.OptionalElse == null)
			return executor.None(iff.ReturnType);
		var elseResult = executor.RunExpression(iff.OptionalElse, ctx);
		return iff.OptionalElse is MutableReassignment || IsMutableInstanceCall(iff.OptionalElse)
			? executor.None(iff.ReturnType)
			: elseResult;
	}

	private static bool IsMutableInstanceCall(Expression expression) =>
		expression is MethodCall { Instance: VariableCall { IsMutable: true } } ||
		expression is Body { Expressions.Count: 1 } body &&
		IsMutableInstanceCall(body.Expressions[0]);
}