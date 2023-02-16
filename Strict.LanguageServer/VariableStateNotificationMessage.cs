namespace Strict.LanguageServer;

public sealed class VariableStateNotificationMessage
{
	public VariableStateNotificationMessage(Dictionary<int, string> lineTextPair) => LineTextPair = lineTextPair;
	public Dictionary<int, string> LineTextPair { get; }
}