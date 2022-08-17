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
		ReturnType = ParseParametersAndReturnType(type, lines[0].AsSpan(Name.Length));
		bodyLines = GetLines(lines);
	}

	public int TypeLineNumber { get; }
	private readonly ExpressionParser parser;

	private Type ParseParametersAndReturnType(Type type, ReadOnlySpan<char> rest)
	{
		if (rest.Length == 0)
			return GetEmptyReturnType(type);
		var closingBracketIndex = rest.LastIndexOf(')');
		var gotBrackets = closingBracketIndex > 0;
		return gotBrackets && rest.Length == 2
			? throw new EmptyParametersMustBeRemoved(this)
			: rest[0] == ' ' && !gotBrackets
				? Type.GetType(rest[1..].ToString())
				: rest[0] != '(' == gotBrackets || rest.Length < 2
					? throw new InvalidMethodParameters(this, rest.ToString())
					: !gotBrackets
						? type.GetType(rest[1..].ToString())
						: ParseParameters(type, rest, closingBracketIndex);
	}

	private Type ParseParameters(Type type, ReadOnlySpan<char> rest, int closingBracketIndex)
	{
		foreach (var nameAndType in rest[1..closingBracketIndex].
			Split(',', StringSplitOptions.TrimEntries))
			parameters.Add(new Parameter(type, nameAndType.ToString()));
		return closingBracketIndex + 2 < rest.Length
			? Type.GetType(rest[(closingBracketIndex + 2)..].ToString())
			: GetEmptyReturnType(type);
	}

	private Type GetEmptyReturnType(Type type) =>
		Name == From
			? type
			: type.GetType(Base.None);

	public sealed class InvalidMethodParameters : ParsingFailed
	{
		public InvalidMethodParameters(Method method, string rest) : base(method.Type, 0, rest,
			method.Name) { }
	}

	public sealed class EmptyParametersMustBeRemoved : ParsingFailed
	{
		public EmptyParametersMustBeRemoved(Method method) : base(method.Type, 0, "", method.Name) { }
	}

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

	private readonly Stack<Body> bodies = new();
	public readonly IReadOnlyList<Line> bodyLines;

	private void FillLine(string line, int methodLineNumber, IList<Line> lines)
	{
		if (line.Length == 0)
			throw new Type.EmptyLineIsNotAllowed(Type, TypeLineNumber + methodLineNumber);
		var tabs = GetTabs(line);
		PushOrPopBodyBasedOnTabsDepth(tabs);
		CheckIndentation(line, TypeLineNumber + methodLineNumber, tabs);
		lines[methodLineNumber - 1] = new Line(this, tabs, line[tabs..], TypeLineNumber + methodLineNumber,
			bodies.Count > 0
				? bodies.Peek()
				: null);
	}

	private void PushOrPopBodyBasedOnTabsDepth(int tabs)
	{
		if (tabs > previousTabs)
			bodies.Push(new Body(this, bodies.Count > 0
				? bodies.Peek()
				: null));
		else if (tabs < previousTabs)
			bodies.Pop();
		previousTabs = tabs;
	}

	private static int GetTabs(string line)
	{
		var tabs = 0;
		foreach (var t in line)
			if (t == '\t')
				tabs++;
			else
				break;
		return tabs;
	}

	private int previousTabs;

	public sealed record Line(Method Method, int Tabs, string Text, int FileLineNumber,
		Body? Body = null)
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
	public bool IsPublic => char.IsUpper(Name[0]);

	public override Type? FindType(string name, Context? searchingFrom = null) =>
		name == Value
			? Type
			: Type.FindType(name, searchingFrom ?? this);

	public const string Value = nameof(Value);//TODO: has a different meaning in for BlockExpression
	public Body Body
	{
		get
		{
			if (cachedMethodBody != null)
				return cachedMethodBody;
			if (bodyLines.Count > 0)
				return ParseBodyExpressions();
			return cachedMethodBody = new Body(this);
		}
	}
	private Body? cachedMethodBody;

	private Body ParseBodyExpressions()
	{
		cachedMethodBody = bodyLines[0].Body!;
		var expressions = new List<Expression>();
		for (var lineNumber = 0; lineNumber < bodyLines.Count; lineNumber++)
			expressions.Add(ParseMethodLine(bodyLines[lineNumber], ref lineNumber));
		cachedMethodBody.SetAndValidateExpressions(expressions, bodyLines);
		return cachedMethodBody;
	}

	public override string ToString() =>
		Name + parameters.ToBrackets() + (ReturnType.Name == Base.None
			? ""
			: " " + ReturnType.Name);
}