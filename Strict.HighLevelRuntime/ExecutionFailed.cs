using Strict.Language;

namespace Strict.HighLevelRuntime;

public class ExecutionFailed(Method method, string message = "", Exception? inner = null)
	: ParsingFailed(method.Type, method.TypeLineNumber, message, inner);