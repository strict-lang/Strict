namespace Strict.LanguageServer;

//ncrunch: no coverage start
public sealed class TestNotificationMessage(int lineNumber, TestState state, string? uri = null,
	string? expression = null, string? methodName = null, string? message = null,
	string? details = null, double durationMs = 0, string? stackTrace = null,
	string? typeName = null)
{
	public int LineNumber { get; } = lineNumber;
	public TestState State { get; } = state;
	public string? Uri { get; } = uri;
	public string? Expression { get; } = expression;
	public string? MethodName { get; } = methodName;
	public string? TypeName { get; } = typeName;
	public string? Message { get; } = message;
	public string? Details { get; } = details;
	public double DurationMs { get; } = durationMs;
	public string? StackTrace { get; } = stackTrace;
}

public enum TestState
{
	Red,
	Green
}