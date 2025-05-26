namespace Strict.Language;

public sealed class RecursiveCallCausesStackOverflow(Body body) : ParsingFailed(body);