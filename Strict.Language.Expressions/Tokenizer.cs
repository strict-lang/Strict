using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language.Expressions;

public ref struct Token
{
	public Token(ReadOnlySpan<char> text, int startIndexOfOriginalString, TokenType type)
	{
		Text = text;
		StartIndexOfOriginalString = startIndexOfOriginalString;
		Type = type;
	}

	public ReadOnlySpan<char> Text { get; }
	public int StartIndexOfOriginalString { get; }
	public TokenType Type { get; }
}

public enum TokenType
{
	Unary,
	Operator,
	String,
	List
}

public sealed class Tokenizer
{
	public Tokenizer(string input) => this.input = input;
	private readonly string input;

	public IReadOnlyList<string> GetTokens()
	{
		// step 0: prechecking
		CheckForInvalidSpacing(input);
		if (input.Contains("()"))
			throw new InvalidBrackets(input);

		// step 1: no operator, with and without string inside (already works, just has to be done at caller)
		if (input.FindFirstOperator() == null) // Todo: Never happens; We will put it in the caller
			return input.Count(c => c == '\"') % 2 == 0
				? new[] { input }
				: throw new UnterminatedString(input);

		// step 3: always parse all strings first, make all inital tokens sperated by space (ignore), comma (keep for lists), brackets (keep for lists and shunting yard)
		// step 4: reduce lists: ( 1 , 2 ) => merge, difficult case is nested lists (probably right to left)
		// step 5: return tokens, if one just use (5, string, list), if 2 unary method call (operator like - or not, then something), if 3 just normal binary.Parse, all other cases, shunting yard to create full tree (only expressions, no assignment)

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
		if (!input.Contains('\"')) // TODO: Check list
		{
			foreach (var checkToken in input.Split(' '))
			{
				CheckForInvalidSpacing(checkToken);
				var token = checkToken;
				var startsList = token[0] == '(';
				if (startsList)
					inList = true;
				if (inList)
				{
					listElements.Add(checkToken);
				}
				else if (startsList && token[^1] == ')')
					tokens.Add(token);
				else
				{
					if (startsList)
					{
						tokens.Add("(");
						token = token[1..];
					}
					if (token[^1] == ')')
					{
						tokens.Add(token[..^1]);
						token = ")";
					}
					tokens.Add(token);
				}
				if (checkToken.EndsWith(')'))
				{
					if (listElements.Any(t => t.IsOperator()))
					{
						tokens.Add("(");
						tokens.Add(listElements[0][1..]);
						for (var index = 1; index < listElements.Count - 1; index++)
							tokens.Add(listElements[index]);
						tokens.Add(listElements[^1][..^1]);
						tokens.Add( ")");
					}
					else
					{
						tokens.Add(string.Join(' ', listElements));
					}
					inList = false;
				}
			}
			return tokens;
		}
		for (var index = 0; index < input.Length; index++)
			ParseCharacter(ref index);
		if (inString)
			throw new UnterminatedString(input);
		AddAndClearCurrentToken();
		return tokens;
	}

	private void CheckForInvalidSpacing(string text)
	{
		if (text == "" || text != text.Trim())
			throw new InvalidSpacing(input);
	}

	private void ParseCharacter(ref int index)
	{
		var startsList = input[index] == '(';
		if (startsList)
			inList = true;
		if (input[index] == ')')
		{
			currentToken = string.Join("", listElements);
			inList = false;
		}
		else if (inList)
		{
			listElements.Add(input[index].ToString());
		}
		else
		{
			var isQuote = input[index] == '\"';
			if (isQuote && IsDoubleQuote(index))
				currentToken += input[index++];
			else if (isQuote)
				inString = !inString;
			else if (inString || input[index] != ' ')
				currentToken += input[index];
			else
				AddAndClearCurrentToken();
		}
	}

	private string currentToken = "";
	private bool inString;

	private bool IsDoubleQuote(int index) =>
		inString && index + 1 < input.Length && input[index + 1] == '\"';

	private void AddAndClearCurrentToken()
	{
		CheckForInvalidSpacing(currentToken);
		tokens.Add(currentToken);
		currentToken = "";
	}

	private List<string> tokens = new();
	private bool inList;
	private List<string> listElements = new();

	public sealed class UnterminatedString : Exception
	{
		public UnterminatedString(string input) : base(input) { }
	}

	public class InvalidSpacing : Exception
	{
		public InvalidSpacing(string input) : base(input) { }
	}

	public class InvalidBrackets : Exception
	{
		public InvalidBrackets(string input) : base(input) { }
	}
}