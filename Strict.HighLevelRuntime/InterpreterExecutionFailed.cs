using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public class InterpreterExecutionFailed : ParsingFailed
{
	public InterpreterExecutionFailed(Method method, string message = "", Exception? inner = null) :
		this(method, method.TypeLineNumber, message, inner) { }

	internal InterpreterExecutionFailed(Method method, int fileLineNumber, string message,
		Exception? inner = null, bool messageListsCallers = false) : base(messageListsCallers
			? message
			: message + GetClickableStacktraceLine(method.Type, fileLineNumber, ""),
		fileLineNumber, inner) =>
		(MethodName, ListsCallers) = (method.ToString(), messageListsCallers);

	protected InterpreterExecutionFailed(Type returnType, string message) : base(returnType, 0,
		message) =>
		MethodName = string.Empty;

	internal string MethodName { get; }
	/// <summary>
	/// The message already names every Strict caller, bodies let it pass instead of wrapping it.
	/// </summary>
	internal bool ListsCallers { get; }
	internal string Headline =>
		(InnerException != null && InnerException is not InterpreterExecutionFailed
			? InnerException.GetType().Name + ": "
			: "") + GetHeadline(Message);

	internal static string BuildMethodFailureMessage(Method method, int fileLineNumber,
		IEnumerable<Expression> expressions, string headline, ExecutionContext ctx)
	{
		var location = GetClickableStacktraceLocation(method.Type, fileLineNumber, "");
		return headline + Environment.NewLine + "Failed in \"" + method.Type.FullName + "." +
			method.Name + "\":" + Environment.NewLine +
			string.Join(Environment.NewLine, expressions) + location +
			BuildCallerChain(ctx.Parent, location);
	}

	internal static string GetHeadline(string message)
	{
		var newLineIndex = message.IndexOf('\n');
		return newLineIndex == -1
			? message
			: message[..newLineIndex].TrimEnd('\r');
	}

	internal static string BuildContextMessage(Method method, Expression expression,
		ExecutionContext ctx, string message)
	{
		var currentFrame =
			GetClickableStacktraceLine(method.Type, expression.LineNumber, method.ToString());
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
			var fileLineNumber = current.CurrentExpressionLineNumber >= 0
				? current.CurrentExpressionLineNumber
				: current.Method.TypeLineNumber;
			var frame = GetClickableStacktraceLine(current.Method.Type, fileLineNumber,
				current.Method.ToString());
			if (frame == lastFrame)
				continue;
			message += frame;
			lastFrame = frame;
		}
		return message;
	}
}