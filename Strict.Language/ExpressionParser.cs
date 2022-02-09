﻿namespace Strict.Language;

/// <summary>
/// Abstracts the actual expressions and parsing away to the Expressions project.
/// <see cref="Method.Body"/> will call this lazily when it is called the first time, which is
/// usually not happening until use. This improves performance a lot as we almost do no parsing.
/// </summary>
public abstract class ExpressionParser
{
	public abstract Expression Parse(Method method);
	public abstract Expression ParseMethodCall(Type type, string initializationLine);
	public abstract Expression? TryParse(Method method, ref int lineNumber);
	public abstract Expression? TryParse(Method method, string line, ref int lineNumber);
}