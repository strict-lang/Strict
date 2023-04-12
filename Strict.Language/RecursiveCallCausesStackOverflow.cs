namespace Strict.Language;

public sealed class RecursiveCallCausesStackOverflow : ParsingFailed
{
	public RecursiveCallCausesStackOverflow(Body body) : base(body) { }
}