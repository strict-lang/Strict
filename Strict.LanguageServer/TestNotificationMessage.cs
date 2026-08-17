namespace Strict.LanguageServer;

//ncrunch: no coverage start
public sealed class TestNotificationMessage
{
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

public enum TestState
{
	Red,
	Green
}