using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Strict.Language.Expressions;

public record struct Token(int Start, int End, TokenType Type);

public enum TokenType : byte
{
	Unary,
	Operator,
	String,
	List,
	BracketOpen,
	BracketClose
}

public sealed class PhraseTokenizer
{
	public PhraseTokenizer(string input) => this.input = input.AsMemory();
	private readonly ReadOnlyMemory<char> input;
	private readonly List<Token> tokens = new();

	// ReSharper disable once CyclomaticComplexity
	// ReSharper disable once ExcessiveIndentation
	// ReSharper disable once MethodTooLong
	public IReadOnlyList<string> GetTokens()
	{
		// step 0: prechecking
		CheckForInvalidSpacingOrInvalidBrackets();

//this will be moved outside
		var inputSpan = input.Span;
		// step 1: no operator, with and without string inside (already works, just has to be done at caller)
		if (!inputSpan.IsOperator()) // Todo: Never happens; We will put it in the caller
			if (!inputSpan.Contains('\"'))
				//TODO: yield return input;
				return new[] { input.ToString() };
			else
			{
				if (inputSpan.Count('\"') % 2 == 0)
					return new[] { input.ToString() };
				else
					throw new UnterminatedString(input);
			}

		// step 3: always parse all strings first, make all inital tokens sperated by space (ignore), comma (keep for lists), brackets (keep for lists and shunting yard)
		var result = new List<string>();
		foreach (var word in inputSpan.Split())
		{
			result.Add(word.ToString());
		}
		return result;
		//(5 + (1 + 2)) * 2
		//(
		//tokens.Add(new Token(0, 1, TokenType.BracketOpen));
		////var tokenSpan = inputSpan[tokens[0].Start.. tokens[0].End]; This is how we get the value from span
		////5
		//tokens.Add(new Token(2, 3, TokenType.Unary));
		//// +
		//tokens.Add(new Token(4, 5, TokenType.Operator));
		//if (inputSpan.Compare("5 + 2"))
		//{
		//	return tokens.Select(t => input.Span[t.Start.. t.End].ToString()).ToArray();
		//}
		/*
		var dummy = Memory<char>.Empty;
		//(
		tokens.Add(new Token(dummy, 5, TokenType.BracketOpen));
		//1
		tokens.Add(new Token(dummy, 6, TokenType.Unary));
		// +
		tokens.Add(new Token(dummy, 8, TokenType.Operator));
		//2
		tokens.Add(new Token(dummy, 10, TokenType.Unary));
		//)
		tokens.Add(new Token(dummy, 11, TokenType.BracketClose));
		//)
		tokens.Add(new Token(dummy, 12, TokenType.BracketClose));
		// *
		tokens.Add(new Token(dummy, 14, TokenType.Operator));
		//2
		tokens.Add(new Token(dummy, 16, TokenType.Unary));

		//TODO: manual testing usecase
		if (inputSpan.Equals("(5 + (1 + 2)) * 2", StringComparison.Ordinal))
			return tokens.Select(t => t.Text.ToString()).ToArray();
		*/
		// step 4: reduce lists: ( 1 , 2 ) => merge, difficult case is nested lists (probably right to left)
		//nothing

		// step 5: return tokens, if one just use (5, string, list), if 2 unary method call (operator like - or not, then something), if 3 just normal binary.Parse, all other cases, shunting yard to create full tree (only expressions, no assignment)
		//return tokens;

		//  max line lenght of this class is 100

		// ("a", "b", "a + b") + 3  -> no tokens. start
		// (s1, s2, s3) + 3         -> ( s1 , s2 , s3 ) + 3  (9 tokens)
		// l1 + 3                   -> l1 + 3 (3 tokens)
		// tokens
		// binary

		// (1, 2, 3) + "lidjsfljfs"
		// (1, 2, 3) + s1
		// l1 + s1
		// tokens
		// binary

		// (1, 2) + (3, 4) + "5"
		// (1, 2) + (3, 4) + s1
		// l1 + l2 + s1
		// tokens
		// shunting yard
		// 2 nested binary
		/*TODO: old
		if (!inputSpan.Contains('\"')) // TODO: Check list
		{
			SpanLineEnumerator
			foreach (var checkToken in inputSpan.Split(' '))
			{
				CheckForInvalidSpacingOrInvalidBrackets(checkToken);
				var token = checkToken;
				var startsList = token[0] == '(';
				if (startsList)
					inList = true;
				if (inList)
				{
					listElements.Add(checkToken);
				}
				else if (startsList && token[^1] == ')')
					oldTokens.Add(token);
				else
				{
					if (startsList)
					{
						oldTokens.Add("(");
						token = token[1..];
					}
					if (token[^1] == ')')
					{
						oldTokens.Add(token[..^1]);
						token = ")";
					}
					oldTokens.Add(token);
				}
				if (checkToken.EndsWith(')'))
				{
					if (listElements.Any(t => t.IsOperator()))
					{
						oldTokens.Add("(");
						oldTokens.Add(listElements[0][1..]);
						for (var index = 1; index < listElements.Count - 1; index++)
							oldTokens.Add(listElements[index]);
						oldTokens.Add(listElements[^1][..^1]);
						oldTokens.Add(")");
					}
					else
					{
						oldTokens.Add(string.Join(' ', listElements));
					}
					inList = false;
				}
			}
			return oldTokens;
		}
		*/
		//for (var index = 0; index < input.Length; index++)
		//	ParseCharacter(ref index);
		//if (inString)
		//	throw new UnterminatedString(input);
		//AddAndClearCurrentToken();
		//return oldTokens;
	}

	private void CheckForInvalidSpacingOrInvalidBrackets()
	{
		if (input.Length == 0 || input.Length != input.Trim().Length)
			throw new InvalidSpacing(input);
		if (input.Span.Contains("()", StringComparison.Ordinal))
			throw new InvalidBrackets(input);
	}

	// ReSharper disable once ExcessiveIndentation
	// ReSharper disable once MethodTooLong
	private void ParseCharacter(ref int index)
	{
		var inputSpan = input.Span;
		var startsList = inputSpan[index] == '(';
		if (startsList)
			inList = true;
		if (inputSpan[index] == ')')
		{
			currentToken = string.Join("", listElements);
			inList = false;
		}
		else if (inList)
		{
			listElements.Add(inputSpan[index].ToString());
		}
		else
		{
			var isQuote = inputSpan[index] == '\"';
			if (isQuote && IsDoubleQuote(index))
				currentToken += inputSpan[index++];
			else if (isQuote)
				inString = !inString;
			else if (inString || inputSpan[index] != ' ')
				currentToken += inputSpan[index];
			else
				AddAndClearCurrentToken();
		}
	}

	private string currentToken = "";
	private bool inString;

	private bool IsDoubleQuote(int index) =>
		inString && index + 1 < input.Length && input.Span[index + 1] == '\"';

	private void AddAndClearCurrentToken()
	{
		//TODO: CheckForInvalidSpacingOrInvalidBrackets(currentToken);
		oldTokens.Add(currentToken);
		currentToken = "";
	}

	//TODO: removed after Tokens fully work for all tests
	private readonly List<string> oldTokens = new();
	//TODO: just a hack, removed
	private bool inList;
	private readonly List<string> listElements = new();

	public sealed class UnterminatedString : Exception
	{
		public UnterminatedString(ReadOnlyMemory<char> input) : base(input.ToString()) { }
	}

	public class InvalidSpacing : Exception
	{
		public InvalidSpacing(ReadOnlyMemory<char> input) : base(input.ToString()) { }
	}

	public class InvalidBrackets : Exception
	{
		public InvalidBrackets(ReadOnlyMemory<char> input) : base(input.ToString()) { }
	}
}