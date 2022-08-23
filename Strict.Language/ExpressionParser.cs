using System;
using System.Collections.Generic;

namespace Strict.Language;

/// <summary>
/// Abstracts the actual expressions and parsing away to the Expressions project.
/// <see cref="Method.GetBodyAndParseIfNeeded()"/> will call this lazily when it is called the
/// first time, which is not happening until a method is actually used.
/// This improves performance a lot as we almost do no parsing on any code (99%+ is not executed).
/// </summary>
public abstract class ExpressionParser
{
	// ReSharper disable once TooManyArguments
	public abstract Expression ParseAssignmentExpression(Type type,
		ReadOnlySpan<char> initializationLine, int fileLineNumber);

	public abstract Expression ParseLineExpression(Body body, ReadOnlySpan<char> line);
	public abstract Expression ParseExpression(Body body, ReadOnlySpan<char> text);
	public abstract List<Expression> ParseListArguments(Body body, ReadOnlySpan<char> text);
}