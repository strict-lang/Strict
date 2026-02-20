namespace Strict.LanguageServer;

public sealed class VariableStateNotificationMessage(Dictionary<int, string> lineTextPair)
{
	//ncrunch: no coverage start, TODO: missing tests
	public Dictionary<int, string> LineTextPair { get; } = lineTextPair;
}