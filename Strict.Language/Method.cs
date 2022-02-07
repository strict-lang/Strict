using System;
using System.Collections.Generic;

namespace Strict.Language;

/// <summary>
/// Methods are parsed lazily, which speeds up type and package parsing enormously and
/// also provides us with all methods in a type usable in any other method if needed.
/// </summary>
public class Method : Context
{
	public Method(Type type, ExpressionParser parser, IReadOnlyList<string> lines) : base(type,
		GetName(lines[0]))
	{
		this.parser = parser;
		ReturnType = Name == From
			? type
			: type.GetType(Base.None);
		ParseDefinition(lines[0][Name.Length..]);
		bodyLines = GetLines(lines);
		body = new Lazy<MethodBody>(() => (MethodBody)parser.Parse(this));
	}

	private readonly ExpressionParser parser;

	public Expression? TryParse(string input)
	{
		var lineNumber = 1;
		return parser.TryParse(this, input, ref lineNumber);
	}

	public Expression? TryParse(string input, ref int lineNumber) =>
		parser.TryParse(this, input, ref lineNumber);

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
		if (rest == "()")
			throw new EmptyParametersMustBeRemoved();
		if (!rest.StartsWith('(') || !rest.EndsWith(')'))
			throw new InvalidSyntax(rest);
		ParseParameters(rest[1..^1]);
	}

	public const string Returns = "returns";
	public class EmptyParametersMustBeRemoved : Exception { }

	public class InvalidSyntax : Exception
	{
		public InvalidSyntax(string rest) : base(rest) { }
	}

	public void ParseParameters(string parametersText)
	{
		foreach (var nameAndType in parametersText.Split(", "))
			parameters.Add(new Parameter(this, nameAndType));
	}

	public readonly IReadOnlyList<Line> bodyLines;

	public sealed record Line(int Tabs, string Text)
	{
		public override string ToString() => new string('\t', Tabs) + Text;
	}

	/// <summary>
	/// Skips the first method declaration line, then counts and removes the tabs from each line.
	/// </summary>
	private static IReadOnlyList<Line> GetLines(IReadOnlyList<string> methodLines)
	{
		var bodyLines = new Line[methodLines.Count - 1];
		for (var lineIndex = 1; lineIndex < methodLines.Count; lineIndex++)
		{
			var line = methodLines[lineIndex];
			var tabs = 0;
			foreach (var t in line)
				if (t == '\t')
					tabs++;
				else
					break;
			bodyLines[lineIndex - 1] = new Line(tabs, line[tabs..]);
			/*obs
			line.coun
			bodyText.AppendLine([1..]);
		var mainLines = new List<string>();
		var currentLine = new StringBuilder(40);
		for (var index = 0; index < bodyText.Length; index++)
			if (bodyText[index] == '\n' && NextLineIsNotExtraIndented(index, bodyText))
			{
				mainLines.Add(currentLine.ToString());
				currentLine.Clear();
			}
			else if (bodyText[index] != '\r' && (index < bodyText.Length - 1 || bodyText[index] != '\n'))
				currentLine.Append(bodyText[index]);
		// Warning: This adds an extra newline at the last expression, which will
		// be cut off in the MethodExpressionParser.GetMainLines again, see test.
		if (currentLine.Length > 0)
			mainLines.Add(currentLine.ToString());
		return mainLines;
	}

	// not used anymore: public class MethodNameCantBeKeyword : Exception { public MethodNameCantBeKeyword(string methodName) : base(methodName) { } }
	// not used anymore: return name.IsKeyword() && !name.IsKeywordFunction() ? throw new MethodNameCantBeKeyword(name) : name;
	
	private static bool NextLineIsNotExtraIndented(int index, string lines) =>
		index + 1 < lines.Length && lines[index + 1] != '\t';
			*/
		}
		return bodyLines;
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