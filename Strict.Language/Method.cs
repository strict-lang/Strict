using System;
using System.Collections.Generic;

namespace Strict.Language;

/// <summary>
/// Methods are parsed lazily, which speeds up type and package parsing enormously and
/// also provides us with all methods in a type usable in any other method if needed.
/// </summary>
public sealed class Method : Context
{
	public Method(Type type, int typeLineNumber, ExpressionParser parser, IReadOnlyList<string> lines) : base(type,
		GetName(lines[0]))
	{
		TypeLineNumber = typeLineNumber;
		this.parser = parser;
		ReturnType = Name == From
			? type
			: type.GetType(Base.None);
		ParseDefinition(lines[0][Name.Length..]);
		bodyLines = GetLines(lines);
		body = new Lazy<MethodBody>(() => (MethodBody)parser.ParseMethodBody(this));
	}

	public int TypeLineNumber { get; }
	private readonly ExpressionParser parser;

	public Expression? TryParseExpression(Line line, string remainingPartToParse) =>
		parser.TryParseExpression(line, remainingPartToParse);

	public Expression ParseMethodLine(Line line, ref int methodLineNumber) =>
		parser.ParseMethodLine(line, ref methodLineNumber);

	/// <summary>
	/// Simple lexer to just parse the method definition and get all used names and types.
	/// Method code itself is parsed in are more complex (BNF), complete and slow way.
	/// </summary>
	private static string GetName(string firstLine) => firstLine.SplitWordsAndPunctuation()[0];

	public const string From = "from";

	private void ParseDefinition(string rest)
	{
		var returnsIndex = rest.IndexOf(" " + Returns + " ", StringComparison.Ordinal);
		if (returnsIndex >= 0)
		{
			ReturnType = Type.GetType(rest[(returnsIndex + Returns.Length + 2)..]);
			rest = rest[..returnsIndex];
		}
		if (string.IsNullOrEmpty(rest))
			return;
		CheckForInvalidSyntax(rest);
		ParseParameters(rest[1..^1]);
	}

	private static void CheckForInvalidSyntax(string rest)
	{
		if (rest == "()")
			throw new EmptyParametersMustBeRemoved();
		if (!rest.StartsWith('(') || !rest.EndsWith(')'))
			throw new InvalidSyntax(rest);
	}

	public const string Returns = "returns";
	public sealed class EmptyParametersMustBeRemoved : Exception { }

	public sealed class InvalidSyntax : Exception
	{
		public InvalidSyntax(string rest) : base(rest) { }
	}

	public void ParseParameters(string parametersText)
	{
		foreach (var nameAndType in parametersText.Split(", "))
			parameters.Add(new Parameter(this, nameAndType));
	}

	/// <summary>
	/// Skips the first method declaration line, then counts and removes the tabs from each line.
	/// </summary>
	private IReadOnlyList<Line> GetLines(IReadOnlyList<string> methodLines)
	{
		var lines = new Line[methodLines.Count - 1];
		for (var methodLineNumber = 1; methodLineNumber < methodLines.Count; methodLineNumber++)
			FillLine(methodLines[methodLineNumber], methodLineNumber, lines);
		return lines;
	}

	private void FillLine(string line, int methodLineNumber, IList<Line> lines)
	{
		if (line.Length == 0)
			throw new Type.EmptyLineIsNotAllowed(Type, TypeLineNumber + methodLineNumber);
		var tabs = 0;
		foreach (var t in line)
			if (t == '\t')
				tabs++;
			else
				break;
		CheckIndentation(line, TypeLineNumber + methodLineNumber, tabs);
		lines[methodLineNumber - 1] = new Line(this, tabs, line[tabs..], TypeLineNumber + methodLineNumber);
	}

	public readonly IReadOnlyList<Line> bodyLines;

	public sealed record Line(Method Method, int Tabs, string Text, int FileLineNumber)
	{
		public override string ToString() => new string('\t', Tabs) + Text;
	}

	private void CheckIndentation(string line, int lineNumber, int tabs)
	{
		if (tabs is 0 or > 3)
			throw new InvalidIndentation(Type, lineNumber, line, Name);
		if (line.Length - tabs != line.TrimStart().Length)
			throw new Type.ExtraWhitespacesFoundAtBeginningOfLine(Type, lineNumber, line, Name);
		if (line.Length != line.TrimEnd().Length)
			throw new Type.ExtraWhitespacesFoundAtEndOfLine(Type, lineNumber, line, Name);
	}

	public sealed class InvalidIndentation : ParsingFailed
	{
		public InvalidIndentation(Type type, int lineNumber, string line, string method) : base(type,
			lineNumber, method, line) { }
	}

	public Type Type => (Type)Parent;
	public IReadOnlyList<Parameter> Parameters => parameters;
	private readonly List<Parameter> parameters = new();
	public Type ReturnType { get; private set; }
	private readonly Lazy<MethodBody> body;
	public MethodBody Body => body.Value;
	public bool IsPublic => char.IsUpper(Name[0]);

	public override Type? FindType(string name, Context? searchingFrom = null) =>
		name == Base.Other
			? Type
			: Type.FindType(name, searchingFrom ?? this);
}