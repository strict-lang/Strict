using System;

namespace Strict.Language.MethodParsing
{
	public class LineLexer
	{
		public LineLexer(Tokenizer tokens) => this.tokens = tokens;
		private readonly Tokenizer tokens;

		public void Process(string line)
		{
			Tabs = 0;
			previousCharacter = ' ';
			for (Position = 0; Position < line.Length; Position++)
				if (line[Position] == '\t')
					Tabs++;
				else if (Tabs == 0)
					throw new LineMustStartWithTab();
				else
					ParseToken(line[Position]);
			if (word.Length > 0)
				ParseWord();
		}

		public int Tabs { get; private set; }
		public int Position { get; private set; }
		public class LineMustStartWithTab : Exception { }

		private void ParseToken(in char character)
		{
			if (character == ' ' && previousCharacter != ')' || character == '(' || character == ')')
				ParseWord();
			if (character == '(')
				tokens.Add(Token.Open);
			else if (character == ')')
				tokens.Add(Token.Close);
			else if (character != ' ' && previousCharacter != ')')
				word += character;
			previousCharacter = character;
		}

		private void ParseWord()
		{
			if (string.IsNullOrEmpty(word))
				throw new UnexpectedSpaceOrEmptyParenthesisDetected(Position);
			if (word.IsKeyword())
				tokens.Add(Token.FromKeyword(word));
			else if (word.IsOperator())
				tokens.Add(Token.FromOperator(word));
			else if (Token.IsValidNumber(word))
				tokens.Add(Token.FromNumber(word));
			else if (Token.IsValidIdentifier(word))
				tokens.Add(Token.FromIdentifier(word));
			else
				throw new InvalidIdentifierName(word, Position);
			word = "";
		}

		private string word = "";
		private int previousCharacter;

		public class UnexpectedSpaceOrEmptyParenthesisDetected : Exception
		{
			public UnexpectedSpaceOrEmptyParenthesisDetected(in int position) : base(
				"at position: " + position) { }
		}

		public class InvalidIdentifierName : Exception
		{
			public InvalidIdentifierName(string word, int position) : base(word + " at position: " +
				position) { }
		}
	}
}