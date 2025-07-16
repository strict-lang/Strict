namespace Strict.Language.Tests;

public sealed class MockRunTypeLines : TypeLines
{
	public MockRunTypeLines(string name = nameof(MockRunTypeLines)) : base(name, "has logger", "Run",
		"\tlog.WriteLine") { }
}