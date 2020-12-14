﻿using System;
using System.Collections.Generic;
using System.Text;

namespace Strict.Language
{
	/// <summary>
	/// Methods are parsed lazily, which speeds up type and package parsing enormously and
	/// also provides us with all methods in a type usable in any other method if needed.
	/// </summary>
	public class Method : Context
	{
		public Method(Type type, ExpressionParser parser, string[] lines) : base(type, GetName(lines[0]))
		{
			ReturnType = Name == From
				? type
				: type.GetType(Base.None);
			ParseDefinition(lines[0].Substring(Name.Length));
			body = new Lazy<MethodBody>(() =>
				(MethodBody)parser.Parse(this, GetMethodBodyLines(lines)));
		}

		/// <summary>
		/// Simple lexer to just parse the method definition and get all used names and types.
		/// Method code itself is parsed in are more complex (BNF), complete and slow way.
		/// </summary>
		private static string GetName(string firstLine)
		{
			var name = firstLine.SplitWordsAndPunctuation()[0];
			return /*unused: name.IsKeyword() && !name.IsKeywordFunction()
				? throw new MethodNameCantBeKeyword(name)
				: */name;
		}

		public const string From = "from";

		/// <summary>
		/// Skip the first method declaration line and remove the first tab from each line.
		/// </summary>
		private static string GetMethodBodyLines(string[] lines)
		{
			var bodyText = new StringBuilder();
			for (int lineIndex = 1; lineIndex < lines.Length; lineIndex++)
				bodyText.AppendLine(lines[lineIndex].Substring(1));
			return bodyText.ToString();
		}

		/*unused
		public class MethodNameCantBeKeyword : Exception
		{
			public MethodNameCantBeKeyword(string methodName) : base(methodName) { }
		}
		*/
		private void ParseDefinition(string rest)
		{
			var returnsIndex = rest.IndexOf(" " + Returns + " ", StringComparison.Ordinal);
			if (returnsIndex >= 0)
			{
				ReturnType = Type.GetType(rest.Substring(returnsIndex + Returns.Length + 2));
				rest = rest.Substring(0, returnsIndex);
			}
			if (string.IsNullOrEmpty(rest))
				return;
			if (rest == "()")
				throw new EmptyParametersMustBeRemoved();
			if (!rest.StartsWith('(') || !rest.EndsWith(')'))
				throw new InvalidSyntax(rest);
			ParseParameters(rest[1..^1]);
		}
		public const string Returns = "returns";

		public class EmptyParametersMustBeRemoved : Exception { }

		public void ParseParameters(string parametersText)
		{
			foreach (var nameAndType in parametersText.Split(", "))
				parameters.Add(new Parameter(this, nameAndType));
		}

		public Type Type => (Type)Parent;

		public class InvalidSyntax : Exception
		{
			public InvalidSyntax(string rest) : base(rest) { }
		}

		public IReadOnlyList<Parameter> Parameters => parameters;
		private readonly List<Parameter> parameters = new List<Parameter>();
		public Type ReturnType { get; private set; }
		private readonly Lazy<MethodBody> body;
		public MethodBody Body => body.Value;

		public override Type? FindType(string name, Context? searchingFrom = null) =>
			name == Base.Other
				? Type
				: Type.FindType(name, searchingFrom ?? this);

		public Member FindMember(string methodName)
		{
			//TODO
			return null!;
		}
	}
}