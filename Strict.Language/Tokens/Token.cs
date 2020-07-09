using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text.RegularExpressions;

namespace Strict.Language.Tokens
{
	public class Token
	{
		/// <summary>
		/// Very restricted, if a token exists already, we want to reuse the exact same one, this is true
		/// for any number, keyword and even identifier, they all will return the same token if the value
		/// or identifier name is the same. It makes parsing much easier as well without value checking.
		/// </summary>
		private Token(string name, object? value = null)
		{
			Name = name;
			Value = value;
		}

		public string Name { get; }
		public object? Value { get; }
		public static bool IsValidNumber(string word) => double.TryParse(word, out _);

		public static Token FromNumber(string word) =>
			FromNumber(Convert.ToDouble(word, CultureInfo.InvariantCulture));

		public static Token FromNumber(double value)
		{
			if (CachedNumbers.TryGetValue(value, out var existing))
				return existing;
			var newNumber = new Token(Number, value);
			CachedNumbers.Add(value, newNumber);
			return newNumber;
		}
		
		public const string Number = nameof(Number);
		private static readonly Dictionary<double, Token> CachedNumbers = new Dictionary<double, Token>();

		/// <summary>
		/// Identifier words must have at least 3 characters, no numbers or special characters in them
		/// and follow camelCase for private members and PascalCase for accessing public members or types.
		/// </summary>
		public static bool IsValidIdentifier(string word) =>
			word.Length >= 3 && (IsPrivateIdentifier(word) || IsPublicIdentifier(word));

		private static bool IsPublicIdentifier(string word) => Regex.IsMatch(word, Public);
		private static string Public => "^[A-Z][a-z]+(?:[A-Z][a-z]+)*$";
		private static bool IsPrivateIdentifier(string word) => Regex.IsMatch(word, Private);
		private static string Private => "^[a-z][a-z]+(?:[A-Z][a-z]+)*$";

		public static Token FromIdentifier(string name)
		{
			if (CachedIdentifiers.TryGetValue(name, out var existing))
				return existing;
			var newIdentifier = new Token(IsPublicIdentifier(name)
				? nameof(Public)
				: nameof(Private), name);
			CachedIdentifiers.Add(name, newIdentifier);
			return newIdentifier;
		}

		private static readonly Dictionary<string, Token> CachedIdentifiers = new Dictionary<string, Token>();

		public override string ToString() =>
			Value != null
				? Value.ToString()!
				: Name;

		public static Token FromKeyword(string keyword) =>
			keyword switch
			{
				Keyword.Test => Test,
				Keyword.Is => Is,
				Keyword.From => From,
				Keyword.Let => Let,
				Keyword.Return => Return,
				Keyword.True => True,
				_ => throw new InvalidKeyword(keyword)
			};
		
		public static Token Test = new Token(Keyword.Test);
		public static Token Is = new Token(Keyword.Is);
		public static Token From = new Token(Keyword.From);
		public static Token Let = new Token(Keyword.Let);
		public static Token Return = new Token(Keyword.Return);
		public static Token True = new Token(Keyword.True);

		public class InvalidKeyword : Exception
		{
			public InvalidKeyword(string keyword) : base(keyword) { }
		}
		
		public static Token FromOperator(string operatorSymbol) =>
			operatorSymbol switch
			{
				Operator.Plus => Plus,
				Operator.Open => Open,
				Operator.Close => Close,
				Operator.Assign => Assign,
				_ => throw new InvalidOperator(operatorSymbol)
			};
		public static Token Plus = new Token(Operator.Plus);
		public static Token Open = new Token(Operator.Open);
		public static Token Close = new Token(Operator.Close);
		public static Token Assign = new Token(Operator.Assign);

		public class InvalidOperator : Exception
		{
			public InvalidOperator(string operatorSymbol) : base(operatorSymbol) { }
		}

		public static Token Dot = new Token(".");

		public static Token FromText(string name)
		{
			if (CachedTexts.TryGetValue(name, out var existing))
				return existing;
			var newText = new Token(Text, name);
			CachedTexts.Add(name, newText);
			return newText;
		}
		
		public const string Text = nameof(Text);
		private static readonly Dictionary<string, Token> CachedTexts = new Dictionary<string, Token>();
	}
}