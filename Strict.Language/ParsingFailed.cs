using System;

namespace Strict.Language;

/// <summary>
/// Every parsing error with details about the type, line number and method used should use this.
/// It is very important to keep the same formatting (at method x in path:line) to make the stack
/// trace output clickable in Visual Studio to directly jump to the place the error happened.
/// </summary>
public class ParsingFailed : Exception
{
	// ReSharper disable once TooManyDependencies
	public ParsingFailed(Type type, int fileLineNumber, string message = "", string method = "",
		Exception? inner = null) : base((message == ""
		? ""
		: message) + "\n   at " + (method == ""
		? type
		: method) + " in " + type.FilePath + ":line " + (fileLineNumber + 1), inner) { }

	protected ParsingFailed(Method.Line line, string? part = null,
		Type? referencingOtherType = null) : this(line.Method.Type, line.FileLineNumber,
		(part ?? line.ToString()) + (referencingOtherType != null
			? " in " + referencingOtherType
			: ""), line.Method.ToString()) { }
}