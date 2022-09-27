using System;

namespace Strict.Language;

/// <summary>
/// Every parsing error with details about the type, line number and method used should use this.
/// It is very important to keep the same formatting (at method x in path:line) to make the stack
/// trace output clickable in Visual Studio to directly jump to the place the error happened.
/// </summary>
public class ParsingFailed : Exception
{
	protected ParsingFailed(Type type, int fileLineNumber, string message = "", string method = "") :
		base(message + GetClickableStacktraceLine(type, fileLineNumber, method)) { }

	internal static string GetClickableStacktraceLine(Type type, int fileLineNumber, string method) =>
		"\n   at " + (method == ""
			? type
			: method) + " in " + type.FilePath + ":line " + (fileLineNumber + 1);

	public ParsingFailed(Type type, int fileLineNumber, string message, Exception inner) : base(
		message + GetClickableStacktraceLine(type, fileLineNumber, ""), inner) { }

	protected ParsingFailed(Body body, string? message = null, Type? referencingOtherType = null) :
		this(body.Method.Type, body.Method.TypeLineNumber + body.ParsingLineNumber,
			(string.IsNullOrEmpty(message)
				? body.CurrentLine
				: message) + (referencingOtherType != null
				? " in " + referencingOtherType
				: ""), body.Method.ToString()) { }
}