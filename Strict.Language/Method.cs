using System;
using System.Collections.Generic;
using System.Text;

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
		ReturnType = Name == From
			? type
			: type.GetType(Base.None);
		ParseDefinition(lines[0][Name.Length..]);
		body = new Lazy<MethodBody>(() => (MethodBody)parser.Parse(this, GetMethodBodyLines(lines)));
	}

	/// <summary>
	/// Simple lexer to just parse the method definition and get all used names and types.
	/// Method code itself is parsed in are more complex (BNF), complete and slow way.
	/// </summary>
	private static string GetName(string firstLine) => firstLine.SplitWordsAndPunctuation()[0];
	// not used anymore: return name.IsKeyword() && !name.IsKeywordFunction() ? throw new MethodNameCantBeKeyword(name) : name;

	public const string From = "from";

	/// <summary>
	/// Skip the first method declaration line and remove the first tab from each line.
	/// Warning: This adds an extra newline at the last expression, which will
	/// be cut off in the MethodExpressionParser.GetMainLines again, see test.
	/// </summary>
	private static string GetMethodBodyLines(IReadOnlyList<string> lines)
	{
		var bodyText = new StringBuilder();
		for (var lineIndex = 1; lineIndex < lines.Count; lineIndex++)
			bodyText.AppendLine(lines[lineIndex][1..]);
		return bodyText.ToString();
	}

	// not used anymore: public class MethodNameCantBeKeyword : Exception { public MethodNameCantBeKeyword(string methodName) : base(methodName) { } }

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

	public void ParseParameters(string parametersText)
	{
		foreach (var nameAndType in parametersText.Split(", "))
			parameters.Add(new Parameter(this, nameAndType));
	}

	public Type Type => (Type)Parent;

	public class InvalidSyntax : Exception
	{
		public InvalidSyntax(string rest) : base(rest) { }
	}

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