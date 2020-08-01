using System;

namespace Strict.Tokens
{
	/// <summary>
	/// Goes through each line and gives us basic Tokens for keywords and Identifiers.
	/// See Expressions namespace for full parsing of method bodies.
	/// </summary>
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
				tokens.Add(DefinitionToken.Open);
			else if (character == ')')
				tokens.Add(DefinitionToken.Close);
			else if (character != ' ' && previousCharacter != ')')
				word += character;
			previousCharacter = character;
		}

		private void ParseWord()
		{
			if (string.IsNullOrEmpty(word))
				throw new UnexpectedSpaceOrEmptyParenthesisDetected(Position);
			/*not used
			if (word.IsKeyword())
				tokens.Add(DefinitionToken.FromKeyword(word));
			else if (word.IsOperator())
				tokens.Add(DefinitionToken.FromOperator(word));
			else */if (DefinitionToken.IsValidNumber(word))
				tokens.Add(DefinitionToken.FromNumber(word));
			else if (DefinitionToken.IsValidIdentifier(word))
				tokens.Add(DefinitionToken.FromIdentifier(word));
			else if (word.StartsWith('\"') && word.EndsWith('\"'))
				tokens.Add(DefinitionToken.FromText(word[1..^1]));
			/*also not used here
			else if (word.Contains('.'))
				AddIdentifierParts();
			*/
			else
				throw new InvalidIdentifierName(word, Position);
			word = "";
		}
		/*nah
		private void AddIdentifierParts()
		{
			var split = word.Split('.');
			for (var index = 0; index < split.Length; index++)
			{
				if (index > 0)
					tokens.Add(DefinitionToken.Dot);
				tokens.Add(DefinitionToken.FromIdentifier(split[index]));
			}
		}
		*/
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