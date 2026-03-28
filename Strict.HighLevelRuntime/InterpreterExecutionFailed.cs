using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public class InterpreterExecutionFailed : ParsingFailed
{
	public InterpreterExecutionFailed(Method method, string message = "", Exception? inner = null) : base(
		method.Type, method.TypeLineNumber, message, inner) { }

	protected InterpreterExecutionFailed(Type returnType, string message) : base(returnType, 0, message) { }

	internal static string GetMethodFailureHeader(Method method) =>
		"Failed in \"" + method.Type.FullName + "." + method.Name + "\":";

	internal static string BuildContextMessage(Method method, Expression expression, ExecutionContext ctx,
   string message)
	{
		var currentFrame = GetClickableStacktraceLine(method.Type, expression.LineNumber, method.ToString());
		return message + currentFrame + BuildCallerChain(ctx.Parent, currentFrame);
	}

	internal static string BuildContextMessage(Method method, int fileLineNumber,
    ExecutionContext ctx, string message)
	{
		var currentFrame = GetClickableStacktraceLine(method.Type, fileLineNumber, method.ToString());
		return message + currentFrame + BuildCallerChain(ctx.Parent, currentFrame);
	}

 private static string BuildCallerChain(ExecutionContext? ctx, string previousFrame)
	{
		var message = string.Empty;
    var lastFrame = previousFrame;
		for (var current = ctx; current != null; current = current.Parent)
   {
			var frame = GetClickableStacktraceLine(current.Method.Type, current.Method.TypeLineNumber,
				current.Method.ToString());
			if (frame == lastFrame)
				continue;
			message += frame;
			lastFrame = frame;
		}
		return message;
	}
}