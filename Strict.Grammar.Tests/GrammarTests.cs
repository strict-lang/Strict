using System.IO;
using Eto.Parse;
using Eto.Parse.Grammars;
//using Eto.Parse.Grammars;
using NUnit.Framework;

namespace Strict.Grammar.Tests
{
	public class Tests
	{
		[Test]
		public void StrictGrammarIsValid()
		{
			// Eto.Parse is not very good at catching errors, but it is the best we have currently.
			// I spent an entire night evaluating alternatives and they weren't any better.

			// Read the official grammar from the @"Grammar\Strict.ebnf" file.
			var source = File.ReadAllText(@"..\..\..\..\Grammar\Strict.ebnf");
			var start = "file";

			// Configure Eto.Parse to use a syntax somewhat similar to the one Ben used initially.
			EbnfStyle style = 0;
			// ... Allow range specifications ([A-Z], etc.) in token definitions (:=).
			style |= EbnfStyle.CharacterSets;
			// ... Use terminating cardinality flags: foo? bar* baz+ (necessary to use ranges).
			style |= EbnfStyle.CardinalityFlags;
			// ... Avoid using comma (,) as the rule separator (almost unreadable).
			style |= EbnfStyle.WhitespaceSeparator;
			// ... Allow backslash-escaped characters in strings.
			style |= EbnfStyle.EscapeTerminalStrings;

			// Attempt to parse the grammar to catch some errors.
			var grammar = new EbnfGrammar(style).Build(source, start);

			// Attempt to convert the grammar to code solely for the sake of discovering more errors.
			string code = new EbnfGrammar(style).ToCode(source, start);
		}
	}
}