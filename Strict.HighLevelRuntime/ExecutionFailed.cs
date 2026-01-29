using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public class ExecutionFailed : ParsingFailed
{
	public ExecutionFailed(Method method, string message = "", Exception? inner = null) : base(
		method.Type, method.TypeLineNumber, message, inner) { }

	protected ExecutionFailed(Type returnType, string message) : base(returnType, 0, message) { }
}