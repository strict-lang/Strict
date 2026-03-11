using Strict.Expressions;
using Strict.Language;

namespace Strict.HighLevelRuntime;

internal sealed class IfEvaluator(Interpreter interpreter)
{
	public ValueInstance Evaluate(If iff, ExecutionContext ctx)
	{
		interpreter.Statistics.IfCount++;
		if (interpreter.RunExpression(iff.Condition, ctx).Boolean)
		{
			var thenResult = interpreter.RunExpression(iff.Then, ctx);
			return iff.Then is MutableReassignment || IsMutableInstanceCall(iff.Then)
				? interpreter.noneInstance
				: thenResult;
		}
		if (iff.OptionalElse == null)
			return interpreter.noneInstance;
		var elseResult = interpreter.RunExpression(iff.OptionalElse, ctx);
		return iff.OptionalElse is MutableReassignment || IsMutableInstanceCall(iff.OptionalElse)
			? interpreter.noneInstance
			: elseResult;
	}

	private static bool IsMutableInstanceCall(Expression expression) =>
		expression is MethodCall { Instance: VariableCall { IsMutable: true } } ||
		expression is Body { Expressions.Count: 1 } body &&
		IsMutableInstanceCall(body.Expressions[0]);
}