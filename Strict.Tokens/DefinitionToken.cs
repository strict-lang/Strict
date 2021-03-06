﻿using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text.RegularExpressions;

namespace Strict.Tokens
{
	/// <summary>
	/// Only for types up to method definitions, method bodies are parsed in the Expressions namespace
	/// </summary>
	public class DefinitionToken
	{
		/// <summary>
		/// Very restricted, if a token exists already, we want to reuse the exact same one, this is true
		/// for any number, keyword and even identifier, they all will return the same token if the value
		/// or identifier name is the same. It makes parsing much easier as well without value checking.
		/// </summary>
		private DefinitionToken(string name, object? value = null)
		{
			Name = name;
			Value = value;
		}

		public string Name { get; }
		public object? Value { get; }
		public static bool IsValidNumber(string word) => double.TryParse(word, out _);

		public static DefinitionToken FromNumber(string word) =>
			FromNumber(Convert.ToDouble(word, CultureInfo.InvariantCulture));

		public static DefinitionToken FromNumber(double value)
		{
			if (CachedNumbers.TryGetValue(value, out var existing))
				return existing;
			var newNumber = new DefinitionToken(Number, value);
			CachedNumbers.Add(value, newNumber);
			return newNumber;
		}
		
		public const string Number = nameof(Number);
		private static readonly Dictionary<double, DefinitionToken> CachedNumbers = new Dictionary<double, DefinitionToken>();
		public bool IsNumber => Name == Number;
		public const string From = "from";
		public const string Returns = "returns";

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
		public bool IsIdentifier => Name == nameof(Public) || Name == nameof(Private);

		public static DefinitionToken FromIdentifier(string name)
		{
			if (CachedIdentifiers.TryGetValue(name, out var existing))
				return existing;
			var newIdentifier = new DefinitionToken(IsPublicIdentifier(name)
				? nameof(Public)
				: nameof(Private), name);
			CachedIdentifiers.Add(name, newIdentifier);
			return newIdentifier;
		}

		private static readonly Dictionary<string, DefinitionToken> CachedIdentifiers = new Dictionary<string, DefinitionToken>();

		public override string ToString() =>
			Value != null
				? Value.ToString()!
				: Name;
		/*not longer needed
		public static DefinitionToken FromKeyword(string keyword) =>
			keyword switch
			{
				Keyword.Test => Test,
				Keyword.Is => Is,
				Keyword.From => From,
				Keyword.Let => Let,
				Keyword.Return => Return,
				Keyword.True => True,
				Keyword.False => False,
				_ => throw new InvalidKeyword(keyword)
			};
		
		public static DefinitionToken Test = new DefinitionToken(Keyword.Test);
		public static DefinitionToken Is = new DefinitionToken(Keyword.Is);
		public static DefinitionToken From = new DefinitionToken(Keyword.From);
		public static DefinitionToken Let = new DefinitionToken(Keyword.Let);
		public static DefinitionToken Return = new DefinitionToken(Keyword.Return);
		public static DefinitionToken True = new DefinitionToken(Keyword.True, true);
		public static DefinitionToken False = new DefinitionToken(Keyword.False, false);
		public bool IsBoolean => Name == Keyword.True || Name == Keyword.False;

		public class InvalidKeyword : Exception
		{
			public InvalidKeyword(string keyword) : base(keyword) { }
		}
		
		public static DefinitionToken FromOperator(string operatorSymbol) =>
			operatorSymbol switch
			{
				Operator.Plus => Plus,
				Operator.Open => Open,
				Operator.Close => Close,
				Operator.Assign => Assign,
				_ => throw new InvalidOperator(operatorSymbol)
			};
		public static DefinitionToken Plus = new DefinitionToken(Operator.Plus);
		public static DefinitionToken Assign = new DefinitionToken(Operator.Assign);

		public class InvalidOperator : Exception
		{
			public InvalidOperator(string operatorSymbol) : base(operatorSymbol) { }
		}
		public static DefinitionToken Dot = new DefinitionToken(".");
		*/
		public static DefinitionToken Open = new DefinitionToken("(");
		public static DefinitionToken Close = new DefinitionToken(")");

		public static DefinitionToken FromText(string name)
		{
			if (CachedTexts.TryGetValue(name, out var existing))
				return existing;
			var newText = new DefinitionToken(Text, name);
			CachedTexts.Add(name, newText);
			return newText;
		}
		
		public const string Text = nameof(Text);
		private static readonly Dictionary<string, DefinitionToken> CachedTexts = new Dictionary<string, DefinitionToken>();
		public bool IsText => Name == Text;
	}
}