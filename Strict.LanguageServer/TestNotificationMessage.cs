namespace Strict.LanguageServer;

//ncrunch: no coverage start
public sealed class TestNotificationMessage(int lineNumber, TestState state)
{
	public int LineNumber { get; } = lineNumber;
	public TestState State { get; } = state;
}

public enum TestState
{
	Red,
	Green
}