namespace Strict.LanguageServer;

//ncrunch: no coverage start
public sealed class TestNotificationMessage(int lineNumber,
	TestState state,
	string? uri = null,
	string? expression = null,
	string? methodName = null,
	string? message = null,
	string? details = null,
	double durationMs = 0,
	string? stackTrace = null,
	string? typeName = null)
{
	public int LineNumber { get; } = lineNumber;
	public TestState State { get; } = state;
	public string? Uri { get; init; } = uri;
	public string? Expression { get; } = expression;
	public string? MethodName { get; init; } = methodName;
	public string? TypeName { get; init; } = typeName;
	public string? Message { get; init; } = message;
	public string? Details { get; } = details;
	public double? DurationMs { get; init; } = durationMs;
	public string? StackTrace { get; init; } = stackTrace;
	public string? ConsoleOutput { get; init; }
	public string? Expected { get; init; }
	public string? Actual { get; init; }
	public int? MethodsCalled { get; init; }
	public int? LinesCalled { get; init; }
	public int? CallCount { get; init; }
}