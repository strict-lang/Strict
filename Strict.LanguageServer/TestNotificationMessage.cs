namespace Strict.LanguageServer;

public sealed class TestNotificationMessage
{
	public int LineNumber { get; }
	public TestState State { get; }

	public TestNotificationMessage(int lineNumber, TestState state)
	{
		LineNumber = lineNumber;
		State = state;
	}
}

public enum TestState
{
	Red,
	Green
}