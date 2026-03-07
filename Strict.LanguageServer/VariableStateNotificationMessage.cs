namespace Strict.LanguageServer;

public sealed class VariableStateNotificationMessage(Dictionary<int, string> lineTextPair)
{
	//ncrunch: no coverage start
	public Dictionary<int, string> LineTextPair { get; } = lineTextPair;
}