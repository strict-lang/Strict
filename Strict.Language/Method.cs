using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace Strict.Language;

/// <summary>
/// Methods are parsed lazily, which speeds up type and package parsing enormously and
/// also provides us with all methods in a type usable in any other method if needed.
/// </summary>
public sealed class Method : Context
{
	public Method(Type type, int typeLineNumber, ExpressionParser parser, IReadOnlyList<string> lines)
		: base(type, GetName(lines[0]))
	{
		TypeLineNumber = typeLineNumber;
		this.parser = parser;
		var rest = lines[0].AsSpan(Name.Length);
		if (Name == From)
		{
			ReturnType = type;
			ParseParameters(rest);
		}
		else
			ReturnType = ParseReturnTypeAndParseParameters(type, rest);
		bodyLines = GetLines(lines);
		body = new Lazy<MethodBody>(() => (MethodBody)parser.ParseMethodBody(this));
	}

	public int TypeLineNumber { get; }
	private readonly ExpressionParser parser;

	private Type ParseReturnTypeAndParseParameters(Type type, ReadOnlySpan<char> rest)
	{
		if (rest.Length <= 0)
		{
			ParseParameters(rest);
			return type.GetType(Base.None);
		}
		var returnsIndex = rest.IndexOf(ReturnsWithSpaces, StringComparison.Ordinal);
		if (returnsIndex >= 0)
		{
			ParseParameters(rest[..returnsIndex]);
			return Type.GetType(rest[(returnsIndex + Returns.Length + 2)..].ToString());
		}
		ParseParameters(rest);
		return type.GetType(Base.None);
	}

	private void ParseParameters(ReadOnlySpan<char> rest)
	{
		if (rest.Length == 0)
			return;
		if (rest[^1] != ')')
			throw rest[0] != '('
				? new ExpectedReturns(this, rest.ToString())
				: new InvalidMethodParameters(this, rest.ToString());
		if (rest.Length == 2)
			throw new EmptyParametersMustBeRemoved(this);
		foreach (var nameAndType in rest[1..^1].Split(',', StringSplitOptions.TrimEntries))
			parameters.Add(new Parameter(this, nameAndType.ToString()));
	}

	public sealed class InvalidMethodParameters : ParsingFailed
	{
		public InvalidMethodParameters(Method method, string rest) : base(method.Type, 0, rest,
			method.Name) { }
	}

	public sealed class EmptyParametersMustBeRemoved : ParsingFailed
	{
		public EmptyParametersMustBeRemoved(Method method) : base(method.Type, 0, "", method.Name) { }
	}

	private const string ReturnsWithSpaces = " " + Returns + " ";

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public Expression ParseExpression(Line line, Range rangeToParse) =>
		parser.ParseExpression(line, rangeToParse);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public List<Expression> ParseListArguments(Line line, Range range) =>
		parser.ParseListArguments(line, range);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public Expression ParseMethodLine(Line line, ref int methodLineNumber) =>
		parser.ParseMethodLine(line, ref methodLineNumber);

	/// <summary>
	/// Simple lexer to just parse the method definition and get all used names and types. Method code
	/// itself is parsed in are more complex way (Shunting yard/PhraserTokenizer/BNF/etc.) and slower.
	/// Examples: Run\n, Run(number)\n, Run returns Text\n
	/// </summary>
	private static string GetName(ReadOnlySpan<char> firstLine)
	{
		var name = firstLine;
		for (var i = 0; i < firstLine.Length; i++)
			if (firstLine[i] == '(' || firstLine[i] == ' ')
			{
				name = firstLine[..i];
				break;
			}
		if (!name.IsWord() && !name.IsOperator())
			throw new NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(name.ToString());
		return name.ToString();
	}

	public const string From = "from";
	public const string Returns = "returns";

	public sealed class ExpectedReturns : ParsingFailed
	{
		public ExpectedReturns(Method method, string rest) :
			base(method.Type, 0, rest, method.Name) { }
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
	public Type ReturnType { get; }
	private readonly Lazy<MethodBody> body;
	public MethodBody Body => body.Value;
	public bool IsPublic => char.IsUpper(Name[0]);
	/// <summary>
	/// Dictionaries are slow and eats up a lot of memory, only created when needed.
	/// </summary>
	private Dictionary<string, Expression>? variables;

	public void AddVariable(string name, Expression value)
	{
		variables ??= new Dictionary<string, Expression>(StringComparer.Ordinal);
		variables.Add(name, value);
	}

	public Expression? FindVariableValue(ReadOnlySpan<char> searchFor)
	{
		if (variables == null)
			return null;
		foreach (var (name, value) in variables)
			if (searchFor.Equals(name, StringComparison.Ordinal))
				return value;
		return null;
	}

	public override Type? FindType(string name, Context? searchingFrom = null) =>
		name == Base.Other
			? Type
			: Type.FindType(name, searchingFrom ?? this);
}