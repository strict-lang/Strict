namespace Strict.LanguageServer;

//ncrunch: no coverage start
public sealed class TestNotificationMessage
{
	public TestNotificationMessage(int lineNumber, TestState state, string? uri = null,
		string? expression = null, string? methodName = null, string? message = null,
		string? details = null, double durationMs = 0, string? stackTrace = null,
		string? typeName = null)
	{
		LineNumber = lineNumber;
		State = state;
		Uri = uri;
		Expression = expression;
		MethodName = methodName;
		TypeName = typeName;
		Message = message;
		Details = details;
		DurationMs = durationMs;
		StackTrace = stackTrace;
	}
	public int LineNumber { get; init; }
	public TestState State { get; init; }
	public string? Uri { get; init; }
	public string? Expression { get; init; }
	public string? MethodName { get; init; }
	public string? TypeName { get; init; }
	public string? Message { get; init; }
	public string? Details { get; init; }
	public double? DurationMs { get; init; }
	public string? StackTrace { get; init; }
	public string? ConsoleOutput { get; init; }
	public string? Expected { get; init; }
	public string? Actual { get; init; }
	public int? MethodsCalled { get; init; }
	public int? LinesCalled { get; init; }
	public int? CallCount { get; init; }
}
}

public enum TestState
{
	Red,
	Green
}