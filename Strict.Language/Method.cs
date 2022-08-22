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
		this.lines = lines;
		ReturnType = ParseParametersAndReturnType(type, lines[0].AsSpan(Name.Length));
		if (lines.Count > 1)
			methodBody = PreParseBody();
	}

	public int TypeLineNumber { get; }
	private readonly ExpressionParser parser;
	private readonly IReadOnlyList<string> lines;
	private Body? methodBody;

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
	/// Also groups all expressions on the same tabs level into bodies. In case a body has only
	/// a single line (which is most often the case), that only expression is used directly.
	/// </summary>
	private Body PreParseBody(int tabs = 0, Body? parent = null)
	{
		var body = new Body(this, tabs, parent);
		for (methodLineNumber++; methodLineNumber < lines.Count; methodLineNumber++)
			FillLine(body);
		return body;
	}

	private int methodLineNumber;

	private void FillLine(Body body)
	{
		var line = lines[methodLineNumber];
		if (line.Length == 0)
			throw new Type.EmptyLineIsNotAllowed(Type, TypeLineNumber + methodLineNumber);
		var tabs = GetTabs(line);
		PushOrPopBodyBasedOnTabsDepth(body, tabs);
		CheckIndentation(line, TypeLineNumber + methodLineNumber, tabs);
		//TODO: methodLines[methodLineNumber - 1] = new Line(this, tabs, line[tabs..], TypeLineNumber + methodLineNumber, bodies.Count > 0 ? bodies.Peek() : null);
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

	//TODO: still needed?
	private readonly Stack<Body> bodies = new();

	private void PushOrPopBodyBasedOnTabsDepth(Body body, int tabs)
	{
		if (tabs > body.Tabs)
			body.PushNestedBody(PreParseBody(tabs, body));
		/*not needed yo?
		else if (tabs < previousTabs)
			bodies.Pop();
		previousTabs = tabs;
		*/
	}

	//TODO: not really needed, we can just the body we are in: private int previousTabs;
	//TODO: remove, not longer needed, directly merged into Body
	public sealed record Line(Method Method, int Tabs, string Text, int FileLineNumber,
		Body? Body = null)
	{
		public override string ToString() => new string('\t', Tabs) + Text;
	}

	//TODO: dummy, remove!
	public List<Line> bodyLines = new();

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
			cachedMethodBody = null!;
			/*TODO
			cachedMethodBody = bodyLines.Count > 0
				? cachedMethodBody = bodyLines[0].Body!
				: new Body(this);
			var lineNumber = 0;
			if (bodyLines.Count > 0)
				ParseBodyExpressions(cachedMethodBody, ref lineNumber);
			*/
			return cachedMethodBody;
		}
	}
	private Body? cachedMethodBody;

	public void ParseBodyExpressions(Body body, ref int methodLineNumber, int tabLevel = 0)
	{
		var expressions = new List<Expression>();
/*TODO
		for (; methodLineNumber < bodyLines.Count; methodLineNumber++)
		{
			if (bodyLines[methodLineNumber].Tabs < tabLevel)
				break;
			expressions.Add(ParseMethodLine(bodyLines[methodLineNumber], ref methodLineNumber));
		}
		body.SetAndValidateExpressions(expressions, bodyLines);
*/
	}

	public override string ToString() =>
		Name + parameters.ToBrackets() + (ReturnType.Name == Base.None
			? ""
			: " " + ReturnType.Name);
}