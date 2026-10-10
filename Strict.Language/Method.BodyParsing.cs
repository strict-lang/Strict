using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("Strict.Language.Tests")]
[assembly: InternalsVisibleTo("Strict.Validators")]
[assembly: InternalsVisibleTo("Strict.HighLevelRuntime")]
[assembly: InternalsVisibleTo("Strict.Bytecode")]

namespace Strict.Language;

public sealed partial class Method
{
	/// <summary>
	/// Skips the first method declaration line, then counts, and removes the tabs from each line.
	/// Also groups all expressions on the same tabs level into bodies. In case a body has only
	/// a single line (which is most often the case), that only expression is used directly.
	/// </summary>
	private Body PreParseBody(int parentTabs = 1, Body? parent = null)
	{
		var body = new Body(this, parentTabs, parent);
		var startLine = methodLineNumber;
		for (; methodLineNumber < lines.Count; methodLineNumber++)
			if (CheckBodyLine(lines[methodLineNumber], body))
				break;
		body.LineRange = new Range(startLine, Math.Min(methodLineNumber, lines.Count));
		return body;
	}

	private bool CheckBodyLine(string line, Body body)
	{
		if (line.Length == 0)
			throw new TypeParser.EmptyLineIsNotAllowed(Type, TypeLineNumber + methodLineNumber);
		var tabs = GetTabs(line);
		if (tabs > body.Tabs)
			PreParseBody(tabs, body);
		CheckIndentation(line, TypeLineNumber + methodLineNumber, tabs);
		return IsCurrentLineInBodyScope(body.Tabs);
	}

	private static int GetTabs(string line)
	{
		var tabs = 0;
		// ReSharper disable once ForCanBeConvertedToForeach, would consume too much memory!
		for (var index = 0; index < line.Length; index++)
			if (line[index] == '\t')
				tabs++;
			else
				break;
		return tabs;
	}

	private void CheckIndentation(string line, int lineNumber, int tabs)
	{
		if (tabs is 0 or > 3)
			throw new InvalidIndentation(Type, lineNumber, line, Name);
		if (tabs == line.Length)
			throw new TypeParser.EmptyLineIsNotAllowed(Type, lineNumber);
		if (char.IsWhiteSpace(line[tabs]))
			throw new TypeParser.ExtraWhitespacesFoundAtBeginningOfLine(Type, lineNumber, line, Name);
		if (char.IsWhiteSpace(line[^1]))
			throw new TypeParser.ExtraWhitespacesFoundAtEndOfLine(Type, lineNumber, line, Name);
	}

	public sealed class InvalidIndentation(Type type, int lineNumber, string line, string method)
		: ParsingFailed(type, lineNumber, method, line);

	private bool IsCurrentLineInBodyScope(int bodyTabs) =>
		methodLineNumber < lines.Count && GetTabs(lines[methodLineNumber]) != bodyTabs;

	private Expression ParseTestsOnlyForGeneric()
	{
		if (methodBody == null)
			throw new CannotCallBodyOnTraitMethod(Type, Name); //ncrunch: no coverage
		if (methodBody.Expressions.Count > 0)
			return methodBody.Expressions.Count == 1
				? methodBody.Expressions[0]
				: methodBody;
		var expressions = new List<Expression>();
		try
		{
			AddDeclarationsAndTests(methodBody, expressions);
		}
		catch (Exception ex) when (ex is not ParsingFailed)
		{
			throw methodBody.FailedAtCurrentLine(ex);
		}
		expressions.Add(new PlaceholderExpression(ReturnType));
		methodBody.SetExpressions(expressions);
		return methodBody;
	}

	private void AddDeclarationsAndTests(Body body, List<Expression> expressions)
	{
		var lastExecutableLineIndex = GetLastExecutableLineIndex();
		for (var index = 1; index < lines.Count; index++)
		{
			var line = lines[index];
			body.ParsingLineNumber = index;
			if (IsDeclarationLine(line))
			{
				expressions.Add(Parser.ParseLineExpression(body, line.AsSpan(body.Tabs)));
				continue;
			}
			if (index == lastExecutableLineIndex || !IsPotentialTestLine(line) || IsControlFlowLine(line))
				continue;
			Expression expression;
			try
			{
				expression = Parser.ParseLineExpression(body, line.AsSpan(body.Tabs));
			}
			catch (Type.GenericTypesCannotBeUsedDirectlyUseImplementation)
			{
				continue;
			}
			if (IsStandaloneInlineTestExpression(expression))
			{
				Tests.Add(expression);
				expressions.Add(expression);
			}
		}
	}

	private int GetLastExecutableLineIndex()
	{
		var lastExecutableLineIndex = -1;
		for (var index = 1; index < lines.Count; index++)
		{
			var line = lines[index];
			if (!line.StartsWith("\t", StringComparison.Ordinal) || line.Length <= 1 ||
				IsControlFlowLine(line))
				continue;
			lastExecutableLineIndex = index;
		}
		return lastExecutableLineIndex;
	}

	private static bool IsPotentialTestLine(string line) =>
		line.Contains($" {BinaryOperator.Is} ", StringComparison.Ordinal) && !line.Contains("?");

	private static bool IsDeclarationLine(string line) =>
		line.StartsWith("\t" + Keyword.Constant + " ", StringComparison.Ordinal) ||
		line.StartsWith("\t" + Keyword.Let + " ", StringComparison.Ordinal) ||
		line.StartsWith("\t" + Keyword.Mutable + " ", StringComparison.Ordinal);

	private static bool IsControlFlowLine(string line) =>
		line.StartsWith("\tif ", StringComparison.Ordinal) ||
		line.StartsWith("\tfor ", StringComparison.Ordinal) ||
		line.StartsWith("\treturn ", StringComparison.Ordinal) ||
		line.StartsWith("\t\t", StringComparison.Ordinal);

	private static bool IsStandaloneInlineTestExpression(Expression expression) =>
		expression.ReturnType.IsBoolean && expression.GetType().Name is not "If" &&
		expression.GetType().Name is not "Return" &&
		expression.GetType().Name is not Body.Declaration &&
		expression.GetType().Name is not Body.MutableReassignment;

	internal sealed class PlaceholderExpression(Type returnType) : Expression(returnType)
	{
		public override bool IsConstant => true; //ncrunch: no coverage
		public override string ToString() => ReturnType.Name;
		public override int GetHashCode() => ReturnType.GetHashCode(); //ncrunch: no coverage

		public override bool Equals(Expression? other) =>
			ReferenceEquals(this, other) || //ncrunch: no coverage
			(other is PlaceholderExpression p && ReturnType == p.ReturnType);
	}

	public string[] GetLinesAndStripTabs(Range innerBodyRange, Body bodyForTabs)
	{
		var result = new string[innerBodyRange.End.Value - innerBodyRange.Start.Value];
		for (var lineNumber = innerBodyRange.Start.Value; lineNumber < innerBodyRange.End.Value;
			lineNumber++)
			result[lineNumber - innerBodyRange.Start.Value] = lines[lineNumber][bodyForTabs.Tabs..];
		return result;
	}
}
