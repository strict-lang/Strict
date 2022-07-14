using System;

namespace Strict.Language;

/// <summary>
/// Every parsing error with details about the type, line number and method used should use this.
/// It is very important to keep the same formatting (at method x in path:line) to make the stack
/// trace output clickable in Visual Studio to directly jump to the place the error happened.
/// </summary>
public class ParsingFailed : Exception
{
	public ParsingFailed(Type type, int fileLineNumber, string message = "", string method = "") :
		base(message + GetClickableStacktraceLine(type, fileLineNumber, method)) { }

	private static string GetClickableStacktraceLine(Type type, int fileLineNumber, string method) =>
		"\n   at " + (method == ""
			? type
			: method) + " in " + type.FilePath + ":line " + (fileLineNumber + 1);

	public ParsingFailed(Type type, int fileLineNumber, string message, Exception inner) : base(
		message + GetClickableStacktraceLine(type, fileLineNumber, ""), inner) { }

	protected ParsingFailed(Method.Line line, string? message = null,
		Type? referencingOtherType = null) : this(line.Method.Type, line.FileLineNumber,
		(message ?? line.ToString()) + (referencingOtherType != null
			? " in " + referencingOtherType
			: ""), line.Method.ToString()) { }
}