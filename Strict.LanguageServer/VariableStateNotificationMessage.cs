namespace Strict.LanguageServer;

public sealed class VariableStateNotificationMessage
{
	public VariableStateNotificationMessage(Dictionary<int, string> lineTextPair) => LineTextPair = lineTextPair;
	//ncrunch: no coverage start, TODO: missing tests
	public Dictionary<int, string> LineTextPair { get; }
}