using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace Strict.Language;

/// <summary>
/// Methods are parsed lazily, which speeds up type and package parsing enormously and
/// also provides us with all methods in a type usable in any other method if needed.
/// </summary>
public sealed class Method : Context
{
	//json parser for comparison how to do async reading lines
	public async Task ParseJson(StreamReader reader)
	{
		var dummy = new Memory<char>(new char[1024]);
		var actualRead = await reader.ReadAsync(dummy);
		dummy.Span.Slice(1, 10);
	}

	public Method(Type type, int typeLineNumber, ExpressionParser parser,
		// Memory<char> lines https://deltaengine.fogbugz.com/f/cases/25240
		IReadOnlyList<string> lines)
		: base(type,
		GetName(lines[0]))
	{
		if (!Name.IsWordWithNumber() && !Name.IsOperator())
			throw new NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(Name);
		TypeLineNumber = typeLineNumber;
		this.parser = parser;
		var rest = lines[0][Name.Length..];
		ReturnType = Name == From
			? type
			: GetReturnType(type, ref rest);
		ParseDefinition(rest);
		bodyLines = GetLines(lines); //https://deltaengine.fogbugz.com/f/cases/25240
		body = new Lazy<MethodBody>(() => (MethodBody)parser.ParseMethodBody(this));
	}

	private void ParseDefinition(string rest)
	{
		if (string.IsNullOrEmpty(rest))
			return;
		if (rest == "()")
			throw new EmptyParametersMustBeRemoved(this);
		ParseParameters(rest[1..^1]);
	}

	private Type GetReturnType(Context type, ref string rest)
	{
		var returnType = type.GetType(Base.None);
		var returnsIndex = rest.IndexOf(" " + Returns + " ", StringComparison.Ordinal);
		if (returnsIndex >= 0)
		{
			returnType = Type.GetType(rest[(returnsIndex + Returns.Length + 2)..]);
			rest = rest[..returnsIndex];
		}
		else if (rest.Split(')').Last().Length > 0)
			throw new ExpectedReturns(this, rest);
		return returnType;
	}

	public int TypeLineNumber { get; }
	private readonly ExpressionParser parser;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public Expression? TryParseExpression(Line line, Range remainingPartToParse) =>
		parser.TryParseExpression(line, remainingPartToParse);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public Expression ParseMethodLine(Line line, ref int methodLineNumber) =>
		parser.ParseMethodLine(line, ref methodLineNumber);

	/// <summary>
	/// Simple lexer to just parse the method definition and get all used names and types. Method code
	/// itself is parsed in are more complex way (Shunting yard/PhraserTokenizer/BNF/etc.) and slower.
	/// </summary>
	private static string GetName(string firstLine)
	{
		//TODO: should be a ReadOnlySpan from the caller
		var block = new Memory<char>(firstLine.ToCharArray());
		var blockSpan = block.Span;
		//Run\n
		//Run(number)
		//Run returns Text\n
		for (var i = 0; i < block.Length; i++)
		{
			if (blockSpan[i] == '(' || blockSpan[i] == ' ' || blockSpan[i] == '\n')
				return blockSpan[..i].ToString();
		}
		//TODO: many times slower
		return firstLine.SplitWordsAndPunctuation()[0];
	}

	public const string From = "from";
	public const string Returns = "returns";

	public sealed class EmptyParametersMustBeRemoved : ParsingFailed
	{
		public EmptyParametersMustBeRemoved(Method method) : base(method.Type, 0, "", method.Name) { }
	}

	public sealed class ExpectedReturns : ParsingFailed
	{
		public ExpectedReturns(Method method, string rest) :
			base(method.Type, 0, rest, method.Name) { }
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
		//TODO: use ReadOnlySpan till here, and then convert to string inside
		lines[methodLineNumber - 1] = new Line(this, tabs, line[tabs..], TypeLineNumber + methodLineNumber);
	}

	public readonly IReadOnlyList<Line> bodyLines;

	/*TODO: just an idea, probably makes no big difference as lines are parsed lazily, so this would keep file memory buffer around, which we also don't like 
	public sealed record NewLine(Method Method, int Tabs, Memory<char> Text, int FileLineNumber)
	{
		public override string ToString() => new string('\t', Tabs) + Text;
	}*/

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
	public readonly List<Expression> Variables = new();

	public override Type? FindType(string name, Context? searchingFrom = null) =>
		name == Base.Other
			? Type
			: Type.FindType(name, searchingFrom ?? this);
}