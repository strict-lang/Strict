using Strict.Language;

namespace Strict.Expressions;

public partial class MethodExpressionParser
{
	/// <summary>
	/// Figures out if there are any bracket groups or if there is a binary expression going on.
	/// Could also contain strings, we don't know. Most of the time it will just be some values.
	/// <see cref="ShuntingYard" /> will parse till the next comma, has to call this till the end.
	/// </summary>
	public override List<Expression> ParseListArguments(Body body, ReadOnlySpan<char> innerSpan)
	{
		if (innerSpan.Contains('(') || (innerSpan.Contains('"') &&
			(innerSpan.Contains(',') || innerSpan.Contains(' '))))
			return If.CanTryParseConditional(body, innerSpan)
				? [If.ParseConditional(body, innerSpan)]
				: new ExpressionListParser(this, innerSpan.ToString()).GetAll(body);
		return innerSpan.Length == 0
			? throw new List.EmptyListNotAllowed(body)
			: ParseAllElementsFast(body, innerSpan, new RangeEnumerator(innerSpan, ',', 0));
	}

	/// <summary>
	/// Similar to TryParseExpression, but we know there are commas separating expressions
	/// </summary>
	public class ExpressionListParser(MethodExpressionParser parser, string inner)
	{
		private readonly ShuntingYard postfix = new(inner);

		/// <summary>
		/// The postfix data comes in upside down, so use another stack to restore order
		/// </summary>
		public List<Expression> GetAll(Body body)
		{
			var expressions = new Stack<Expression>();
			if (postfix.Output.Count == 1)
				expressions.Push(parser.ParseTextWithSpacesOrListWithMultipleOrNestedElements(body,
					inner[postfix.Output.Pop()]));
			else if (postfix.Output.Count == 2)
				expressions.Push(
					parser.ParseMethodCallWithArguments(body, inner.AsSpan(),
						postfix)); //ncrunch: no coverage
			else
				ParseBinaryOrNormalExpressionsIntoList(body, expressions);
			return [.. expressions];
		}

		private void ParseBinaryOrNormalExpressionsIntoList(Body body, Stack<Expression> expressions)
		{
			do
			{
				var span = inner[postfix.Output.Peek()];
				try
				{
					// Is this a binary expression we have to put into the list (tokenized and postfixed)?
					expressions.Push((span.Length == 1 && span[0].IsSingleCharacterOperator()) ||
						span.IsMultiCharacterOperator()
							? Binary.Parse(body, inner.AsSpan(), postfix.Output)
							: body.Method.ParseExpression(body, inner[postfix.Output.Pop()]));
				}
				catch (UnknownExpression ex)
				{
					throw new UnknownExpressionForArgument(body,
						span + " is invalid for argument " + expressions.Count + " " + ex.Message);
				}
				if (postfix.Output.Count > 0 && inner[postfix.Output.Pop().Start.Value] != ',')
					throw new ListTokensAreNotSeparatedByComma(body);
			} while (postfix.Output.Count > 0);
		}
	}

	private static List<Expression> ParseAllElementsFast(Body body, ReadOnlySpan<char> input,
		RangeEnumerator elements)
	{
		var expressions = new List<Expression>();
		foreach (var element in elements)
			try
			{
				expressions.Add(body.Method.ParseExpression(body, input[element]));
			}
			catch (UnknownExpression ex)
			{
				throw new UnknownExpressionForArgument(body,
					input[element].ToString() + " (argument " + expressions.Count + ")\n" + ex.StackTrace);
			}
		return expressions;
	}
}
