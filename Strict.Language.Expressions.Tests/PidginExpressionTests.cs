using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using NUnit.Framework;
using Pidgin;
using Pidgin.Expression;
using static Pidgin.Parser;
using static Pidgin.Parser<char>;

namespace Strict.Language.Expressions.Tests
{
	/// <summary>
	/// Based on https://github.com/benjamin-hodgson/Pidgin/blob/master/Pidgin.Examples/Expression/ExprParser.cs
	/// Trying to make it more strict and figure out the ways to call Pidgin
	/// </summary>
	public class PidginExpressionTests
	{
		[Test]
		public void TestWhitespaceAtBeginningNotAllowed()
		{
			var exception = Assert.Throws<Exception>(() =>
				ParseJson("      [ { \"foo\" : \"bar\" } , [ \"baz\" ] ]"));
			Assert.That(exception.Message, Contains.Substring("unexpected whitespace"));
		}

		private static IJson ParseJson(string input)
		{
			try
			{
				var result = JsonParser.Parse(input);
				Assert.That(result.ToString(), Is.EqualTo(ExpectedJson));
				return result;
			}
			catch (ParseException ex)
			{
				throw new Exception(ex.Message.Replace("unexpected  ", "unexpected whitespace"),
					ex.InnerException);
			}
		}

		private static string ExpectedJson => "[{\"foo\" : \"bar\"}, [\"baz\"]]";

		[Test]
		public void TestWhitespaceInMiddleNotAllowed()
		{
			var exception = Assert.Throws<Exception>(() =>
				ParseJson("[ { \"foo\" : \"bar\" } , [ \"baz\" ] ]"));
			Assert.That(exception.Message, Contains.Substring("unexpected whitespace"));
		}

		[Test]
		public void TestJsonWithoutWhitespaces() =>
			Assert.That(ParseJson(ExpectedJson).ToString(), Is.EqualTo(ExpectedJson));

		[Test]
		public void TestExpression()
		{
			var input = "12 * 3                  ";
			var expression = PidginExpressionParser.ParseOrThrow(input);
			Console.WriteLine(expression);
			Assert.That(expression,
				Is.EqualTo(new BinaryOp(BinaryOperatorType.Mul, new Literal(12), new Literal(3))));
			/*TODO
			var input = "12 * 3 + foo(-3, x)() * (2 + 1)";
			var expected = new BinaryOp(BinaryOperatorType.Add,
				new BinaryOp(BinaryOperatorType.Mul, new Literal(12), new Literal(3)),
				new BinaryOp(BinaryOperatorType.Mul,
					new Call(
						new Call(new Identifier("foo"),
							ImmutableArray.Create<IExpr>(new UnaryOp(UnaryOperatorType.Neg, new Literal(3)),
								new Identifier("x"))), ImmutableArray.Create<IExpr>()),
					new BinaryOp(BinaryOperatorType.Add, new Literal(2), new Literal(1))));
			Assert.That(PidginExpressionParser.ParseOrThrow(input), Is.EqualTo(expected));
			*/
		}
	}
	public static class JsonParser
	{
		public static IJson Parse(string input) => Json.ParseOrThrow(input);
		
		private static readonly Parser<char, char> Quote = Char('"');
		//private static readonly Parser<char, char> Comma = Char(',');
		private static readonly Parser<char, char> CommaWithSpace = Char(',').Then(Char(' '));
		private static readonly Parser<char, char> LBracket = Char('[');
		private static readonly Parser<char, char> RBracket = Char(']');
		//private static readonly Parser<char, char> Colon = Char(':');
		private static readonly Parser<char, char> ColonWithSpaces =
			Char(':').Between(Char(' '), Char(' '));
		private static readonly Parser<char, char> LBrace = Char('{');
		private static readonly Parser<char, char> RBrace = Char('}');

		private static readonly Parser<char, string> String =
			Token(c => c != '"')
				.ManyString()
				.Between(Quote);
		private static readonly Parser<char, IJson> JsonString =
			String.Select<IJson>(s => new JsonString(s));
            
		private static readonly Parser<char, IJson> Json =
			JsonString.Or(Rec(() => JsonArray)).Or(Rec(() => JsonObject));

		private static readonly Parser<char, IJson> JsonArray = 
			Json//.Between(SkipWhitespaces)
				.Separated(CommaWithSpace)
				.Between(LBracket, RBracket)
				.Select<IJson>(els => new JsonArray(els.ToImmutableArray()));
        
		private static readonly Parser<char, KeyValuePair<string, IJson>> JsonMember =
			String
				.Before(ColonWithSpaces)
				.Then(Json, (name, val) => new KeyValuePair<string, IJson>(name, val));

		private static readonly Parser<char, IJson> JsonObject = 
			JsonMember//.Between(SkipWhitespaces)
				.Separated(CommaWithSpace)
				.Between(LBrace, RBrace)
				.Select<IJson>(kvps => new JsonObject(kvps.ToImmutableDictionary()));
	}
	public interface IJson
	{
	}
    
	public class JsonArray : IJson
	{
		public ImmutableArray<IJson> Elements { get; }
		public JsonArray(ImmutableArray<IJson> elements) => Elements = elements;

		public override string ToString()
			=> $"[{string.Join(", ", Elements.Select(e => e.ToString()))}]";
	}

	public class JsonObject : IJson
	{
		public IImmutableDictionary<string, IJson> Members { get; }
		public JsonObject(IImmutableDictionary<string, IJson> members) => Members = members;

		public override string ToString()
			=> $"{{{string.Join(", ", Members.Select(kvp => $"\"{kvp.Key}\" : {kvp.Value}"))}}}";
	}

	public class JsonString : IJson
	{
		public string Value { get; }
		public JsonString(string value) => Value = value;

		public override string ToString()
			=> $"\"{Value}\"";
	}

	public static class PidginExpressionParser
	{
		private static Parser<char, T> Tok<T>(Parser<char, T> token) => Try(token).Before(SkipWhitespaces);
        private static Parser<char, string> Tok(string token)
            => Tok(String(token));
				//private static Parser<char, T> ExpressionToken<T>(Parser<char, T> token, string label) =>
				//	Try(token).Before(End).Labelled(label);

        private static Parser<char, T> Parenthesized<T>(Parser<char, T> parser)
            => parser.Between(Tok("("), Tok(")"));

        private static Parser<char, Func<IExpr, IExpr, IExpr>> Binary(Parser<char, BinaryOperatorType> op)
            => op.Select<Func<IExpr, IExpr, IExpr>>(type => (l, r) => new BinaryOp(type, l, r));
        private static Parser<char, Func<IExpr, IExpr>> Unary(Parser<char, UnaryOperatorType> op)
            => op.Select<Func<IExpr, IExpr>>(type => o => new UnaryOp(type, o));

        private static readonly Parser<char, Func<IExpr, IExpr, IExpr>> Add
            = Binary(Tok("+").ThenReturn(BinaryOperatorType.Add));
        private static readonly Parser<char, Func<IExpr, IExpr, IExpr>> Mul
            = Binary(Tok("*").ThenReturn(BinaryOperatorType.Mul));
        private static readonly Parser<char, Func<IExpr, IExpr>> Neg
            = Unary(Tok("-").ThenReturn(UnaryOperatorType.Neg));
        private static readonly Parser<char, Func<IExpr, IExpr>> Complement
            = Unary(Tok("~").ThenReturn(UnaryOperatorType.Complement));

        private static readonly Parser<char, IExpr> Identifier
            = Tok(Letter.Then(LetterOrDigit.ManyString(), (h, t) => h + t))
                .Select<IExpr>(name => new Identifier(name))
                .Labelled("identifier");
        private static readonly Parser<char, IExpr> Literal
            = Tok(Num)
                .Select<IExpr>(value => new Literal(value))
                .Labelled("integer literal");

        private static Parser<char, Func<IExpr, IExpr>> Call(Parser<char, IExpr> subExpr)
            => Parenthesized(subExpr.Separated(Tok(",")))
                .Select<Func<IExpr, IExpr>>(args => method => new Call(method, args.ToImmutableArray()))
                .Labelled("function call");

        private static readonly Parser<char, IExpr> Expr = Pidgin.Expression.ExpressionParser.Build<char, IExpr>(
            expr => (
                OneOf(
                    Identifier,
                    Literal,
                    Parenthesized(expr).Labelled(nameof(Parenthesized))
                ),
                new[]
                {
                    Operator.PostfixChainable(Call(expr)),
                    Operator.Prefix(Neg).And(Operator.Prefix(Complement)),
                    Operator.InfixL(Mul),
                    Operator.InfixL(Add)
                }
            )
        ).Labelled("expression");

        public static IExpr ParseOrThrow(string input)
            => Expr.ParseOrThrow(input);
		/*
		//nah: private static Parser<char, T> Tok<T>(Parser<char, T> token) => Try(token).Before(SkipWhitespaces);
		//nah: private static Parser<char, string> Tok(string token) => Tok(String(token));
		public static IExpr ParseOrThrow(string input) => Expression.ParseOrThrow(input);

		private static readonly Parser<char, IExpr> Expression = Pidgin.Expression.ExpressionParser.
			Build<char, IExpr>(expr => (
				OneOf(Identifier, Literal, Parenthesized(expr)),
				new[]
				{
					Operator.PostfixChainable(Call(expr)),
					Operator.Prefix(Neg).And(Operator.Prefix(Complement)),
					Operator.InfixL(Mul),
					Operator.InfixL(Add)
				})).Labelled(nameof(Language.Expression));

		private static readonly Parser<char, Func<IExpr, IExpr, IExpr>> Add =
			Binary(Char('+').ThenReturn(BinaryOperatorType.Add));

		private static Parser<char, Func<IExpr, IExpr, IExpr>>
			Binary(Parser<char, BinaryOperatorType> op) =>
			op.Select<Func<IExpr, IExpr, IExpr>>(type => (l, r) => new BinaryOp(type, l, r));

		private static readonly Parser<char, Func<IExpr, IExpr, IExpr>> Mul =
			Binary(Tok('*').ThenReturn(BinaryOperatorType.Mul));

		private static readonly Parser<char, Func<IExpr, IExpr>> Neg =
			Unary(Tok('-').ThenReturn(UnaryOperatorType.Neg));

		private static Parser<char, Func<IExpr, IExpr>>
			Unary(Parser<char, UnaryOperatorType> op) =>
			op.Select<Func<IExpr, IExpr>>(type => o => new UnaryOp(type, o));

		private static readonly Parser<char, Func<IExpr, IExpr>> Complement =
			Unary(Char('~').ThenReturn(UnaryOperatorType.Complement));

		private static readonly Parser<char, IExpr> Identifier =
			Tok(Letter.Then(LetterOrDigit.ManyString(), (h, t) => h + t),
				nameof(Identifier)).Select<IExpr>(name => new Identifier(name));
		

		private static readonly Parser<char, IExpr> Literal =
			Tok(Num, nameof(Literal)).Select<IExpr>(value => new Literal(value));

		private static Parser<char, Func<IExpr, IExpr>> Call(Parser<char, IExpr> subExpr) =>
			Parenthesized(subExpr.Separated(Char(','))).
				Select<Func<IExpr, IExpr>>(
					args => method => new Call(method, args.ToImmutableArray())).
				Labelled("function call");

		//private static Parser<char, T> Parenthesized<T>(Parser<char, T> parser) =>
		//	parser.Between(ExpressionToken(Char('('), "Open"), ExpressionToken(Char(')'), "Close")).
		//		Labelled(nameof(Parenthesized));
		private static Parser<char, T> Tok<T>(Parser<char, T> token, string name)
			=> Try(token).Before(SkipWhitespaces).Labelled(name);
		private static Parser<char, string> Tok(string token, string name)
			=> Tok(String(token), name);
		private static Parser<char, string> Tok(char token)
			=> Tok(String(token.ToString()), token.ToString());

		private static Parser<char, T> Parenthesized<T>(Parser<char, T> parser)
			=> parser.Between(Tok('('), Tok(')'));
		*/
	}

	public interface IExpr : IEquatable<IExpr> { }

	public class Identifier : IExpr
	{
		public string Name { get; }

		public Identifier(string name) => Name = name;

		public bool Equals(IExpr other) => other is Identifier i && Name == i.Name;

		public override string ToString() => nameof(Identifier) + ": " + Name;
	}

	public class Literal : IExpr
	{
		public int Value { get; }

		public Literal(int value) => Value = value;

		public bool Equals(IExpr other) => other is Literal l && Value == l.Value;

		public override string ToString() => nameof(Literal) + ": " + Value;
	}

	public class Call : IExpr
	{
		public IExpr Expr { get; }
		public ImmutableArray<IExpr> Arguments { get; }

		public Call(IExpr expr, ImmutableArray<IExpr> arguments)
		{
			Expr = expr;
			Arguments = arguments;
		}

		public bool Equals(IExpr other) =>
			other is Call c && Expr.Equals(c.Expr) && Arguments.SequenceEqual(c.Arguments);
	}

	public enum UnaryOperatorType
	{
		Neg,
		Complement
	}

	public class UnaryOp : IExpr
	{
		public UnaryOperatorType Type { get; }
		public IExpr Expr { get; }

		public UnaryOp(UnaryOperatorType type, IExpr expr)
		{
			Type = type;
			Expr = expr;
		}

		public bool Equals(IExpr other) =>
			other is UnaryOp u && Type == u.Type && Expr.Equals(u.Expr);
	}

	public enum BinaryOperatorType
	{
		Add,
		Mul
	}

	public class BinaryOp : IExpr
	{
		public BinaryOperatorType Type { get; }
		public IExpr Left { get; }
		public IExpr Right { get; }

		public BinaryOp(BinaryOperatorType type, IExpr left, IExpr right)
		{
			Type = type;
			Left = left;
			Right = right;
		}

		public bool Equals(IExpr other) =>
			other is BinaryOp b && Type == b.Type && Left.Equals(b.Left) && Right.Equals(b.Right);
	}
}