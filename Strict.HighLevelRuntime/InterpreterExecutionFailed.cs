using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public class InterpreterExecutionFailed : ParsingFailed
{
	public InterpreterExecutionFailed(Method method, string message = "", Exception? inner = null) : base(
		method.Type, method.TypeLineNumber, message, inner) { }

	protected InterpreterExecutionFailed(Type returnType, string message) : base(returnType, 0, message) { }

	internal static string BuildContextMessage(Method method, Expression expression, ExecutionContext ctx,
		string message) => message + GetClickableStacktraceLine(method.Type, expression.LineNumber,
			method.ToString()) + BuildCallerChain(ctx.Parent);

	internal static string BuildContextMessage(Method method, int fileLineNumber,
		ExecutionContext ctx, string message) => message +
		GetClickableStacktraceLine(method.Type, fileLineNumber, method.ToString()) +
		BuildCallerChain(ctx.Parent);

	private static string BuildCallerChain(ExecutionContext? ctx)
	{
		var message = string.Empty;
		for (var current = ctx; current != null; current = current.Parent)
			message += GetClickableStacktraceLine(current.Method.Type, current.Method.TypeLineNumber,
				current.Method.ToString());
		return message;
	}
}