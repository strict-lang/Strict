using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions;

/// <summary>
/// Parses method bodies by splitting into main lines (lines starting without tabs)
/// and getting the expression recursively via parser combinator logic in each expression.
/// </summary>
public class MethodExpressionParser : ExpressionParser
{
	public override Expression ParseAssignmentExpression(Type type, string initializationLine, int fileLineNumber)
	{
		var constructor = type.Methods[0];
		var line = new Method.Line(constructor, 0, initializationLine, fileLineNumber);
		return new MethodCall(new Value(type, type), constructor,
			TryParseExpression(line, ..) ?? throw new Method.UnknownExpression(line));
	}

	public override Expression? TryParseExpression(Method.Line line, Range rangeToParse)
	{
		var input = line.Text.GetSpanFromRange(rangeToParse);
		if (input.IsEmpty)
			throw new CannotParseEmptyInput(line);
		if (!input.Contains(' ') && !input.Contains(','))
		{
			remainingToParse = 0;
			return Boolean.TryParse(line, rangeToParse) ?? Text.TryParse(line, rangeToParse) ??
				List.TryParseWithSingleElement(line, rangeToParse) ??
				MemberCall.TryParseMemberOrZeroOrOneArgumentMethodCall(line, rangeToParse) ??
				Number.TryParse(line, rangeToParse);
		}
		//check for method call/member call here too
		var postfix = new ShuntingYard(line.Text, rangeToParse);
		remainingToParse = postfix.RemainingToParse;//TODO: maybe not longer needed, parses , fine already
		return postfix.Output.Count switch
		{
			1 => TryParseTextWithSpacesOrListWithMultipleOrNestedElements(line, postfix.Output.Pop()),
			2 => // TODO: Unary goes here
				throw new NotSupportedException("Feed Murali more to get Unary done"),
			_ => //TODO: should never happen here, Binary will complain if we have a comma there! postfix.Output.Count % 2 != 1 && line.Text[postfix.Output.Skip(1).First().Start.Value] == ',' ?
				Binary.TryParse(line, postfix.Output)
		};
		/*from list, same
		 
		var postfixTokens = new ShuntingYard(line.Text, range).Output;
		if (postfixTokens.Count == 1)
			elementsToFill.Add(line.Method.TryParseExpression(line, postfixTokens.Pop()) ??
				throw new MethodExpressionParser.UnknownExpression(line, line.Text[range]));
		else
			foreach (var token in postfixTokens)
			{
				var expressions = new List<Expression>();
				new PhraseTokenizer(line.Text, new Range(start, start + innerSpan.Length)).ProcessEachToken(
					tokenRange =>
					{
						Console.WriteLine("TryParseWithMultipleOrNestedElements: token=" +
							line.Text[tokenRange.Start.Value]);
						if (line.Text[tokenRange.Start.Value] != ',')
							expressions.Add(line.Method.TryParseExpression(line, tokenRange)
								? 7
						throw new MethodExpressionParser.UnknownExpression(line, line.Text[tokenRange]));
					});
			}
		 */
	}

	private int remainingToParse;

	public class CannotParseEmptyInput : ParsingFailed
	{
		public CannotParseEmptyInput(Method.Line line) : base(line) { }
	}

	/// <summary>
	/// Figures out if there are any bracket groups or if there is binary expression action going on.
	/// Could also contain strings, we don't know. Most of the time it will just be a bunch of values.
	/// <see cref="ShuntingYard"/> will only parse till the next comma, has to call this till the end.
	/// </summary>
	public override List<Expression> ParseListArguments(Method.Line line, int start, int end)
	{
		var innerSpan = line.Text.AsSpan(start, end-start);
		if (innerSpan.Contains('(') || innerSpan.Contains('"'))
		{
			// The postfix data comes in upside down, so use another stack to restore order
			var expressions = new Stack<Expression>();
			// Similar to TryParseExpression, but we know there is commas separating things! 
			var postfix = new ShuntingYard(line.Text, new Range(start, end));
			remainingToParse = postfix.RemainingToParse;//TODO: maybe not longer needed, parses , fine already
			if (postfix.Output.Count == 1)
				expressions.Push(TryParseTextWithSpacesOrListWithMultipleOrNestedElements(line,
					postfix.Output.Pop()) ??
					throw new Method.UnknownExpression(line, line.Text[new Range(start, end)]));
			else if (postfix.Output.Count == 2)
				throw new NotSupportedException("Feed Murali more to get Unary done");
			else
				do
				{
					var range = postfix.Output.Peek();
					Console.WriteLine("pushing list element "+line.Text[range]);
					var span = line.Text.GetSpanFromRange(range);
					// Is this a binary expression we have to put into the list (already tokenized and postfixed)
					if (span.Length == 1 && span[0].IsSingleCharacterOperator() ||
						span.IsMultiCharacterOperator())
						expressions.Push(Binary.TryParse(line, postfix.Output) ??
							throw new Method.UnknownExpression(line, line.Text[range]));
					else
						expressions.Push(line.Method.ParseExpression(line, postfix.Output.Pop()));
					if (postfix.Output.Count > 0 && line.Text[postfix.Output.Pop().Start.Value] != ',')
						throw new ListTokensAreNotSeparatedByComma(line);
				} while (postfix.Output.Count > 0);
				//postfix.Output.Count % 2 != 1 &&
				//line.Text[postfix.Output.Skip(1).First().Start.Value] == ',' ?;}
				//do
				//{
				//	start = end - remainingToParse;
				//} while (start < end);
			return expressions.ToList();
		}
		return ParseAllElementsFast(line, (start, innerSpan.Length),
			innerSpan.SplitIntoRanges(',', true));
	}

	public class ListTokensAreNotSeparatedByComma : ParsingFailed
	{
		public ListTokensAreNotSeparatedByComma(Method.Line line) : base(line) { }
	}

	private static List<Expression> ParseAllElementsFast(Method.Line line, (int, int) offsetAndLength, RangeEnumerator elements)
	{
		var expressions = new List<Expression>();
		foreach (var element in elements)
			expressions.Add(line.Method.ParseExpression(line, element.GetOuterRange(offsetAndLength)));
		return expressions;
	}
	
	//TODO: Probably not needed
	public static bool HasIncompatibleDimensions(Expression left, Expression right) =>
		left is List leftList && right is List rightList &&
		leftList.Values.Count != rightList.Values.Count;

	//TODO: as discussed in meeting, we use generics and always check if the right side is castable into the left side (via from), e.g. make a test where we add a Count to a list of Texts -> output list of texts (always from left side), we never change the left side type
	public static bool HasMismatchingTypes(Expression left, Expression right) =>
		left is List leftList && !leftList.IsFirstType<Text>() && right switch
		{
			List rightList when rightList.IsFirstType<Text>() => true,
			Binary { Left: List rightBinaryLeftList } when rightBinaryLeftList.IsFirstType<Text>() =>
				true,
			Binary { Left: Text } => true,
			_ => !leftList.IsFirstType<Text>() && right is Text
		};

	public sealed class ListsHaveDifferentDimensions : ParsingFailed
	{
		public ListsHaveDifferentDimensions(Method.Line line, string error) : base(line, error) { }
	}

	private static Expression?
		TryParseTextWithSpacesOrListWithMultipleOrNestedElements(Method.Line line, Range range) =>
		Text.TryParse(line, range) ?? List.TryParseWithMultipleOrNestedElements(line, range);

	/// <summary>
	/// Called lazily by Method.Body and only if needed for execution (context should be over there
	/// as parsing is done in parallel, we should not keep any state here).
	/// </summary>
	public override Expression ParseMethodBody(Method method)
	{
		var expressions = new List<Expression>();
		for (var lineNumber = 0; lineNumber < method.bodyLines.Count; lineNumber++)
		{
			var expression = ParseMethodLine(method.bodyLines[lineNumber], ref lineNumber);
			if (expression is Assignment assignment)
				method.Variables.Add(assignment);
			expressions.Add(expression);
		}
		//TODO: to clear link to original memory: method.bodyLines = Memory<char>.Empty; //ArraySegment<Method.Line>.Empty;
		return new MethodBody(method, expressions);
	}

	//private readonly PhraseTokenizer tokenizer = new();

	public override Expression ParseMethodLine(Method.Line line, ref int methodLineNumber) =>
		Assignment.TryParse(line) ?? If.TryParse(line, ref methodLineNumber) ??
		//https://deltaengine.fogbugz.com/f/cases/25210
		Return.TryParse(line) ??
		TryParseExpression(line, ..) ?? throw new Method.UnknownExpression(line);
}