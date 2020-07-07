using System;
using System.Collections.Generic;
using Strict.Language.Extensions;

namespace Strict.Language
{
	/// <summary>
	/// Methods are parsed lazily, which speeds up type and package parsing enormously and
	/// also provides us with all methods in a type usable in any other method if needed.
	/// </summary>
	public class Method : Context
	{
		public Method(Type type, string definitionLine, IReadOnlyList<string> bodyLines) : base(type,
			GetName(definitionLine))
		{
			ReturnType = Name == Keyword.From ? type : type.GetType(Base.None);
			ParseDefinition(definitionLine.Substring(Name.Length));
			body = new Lazy<MethodBody>(() => new MethodBody(this, bodyLines));
		}

		/// <summary>
		/// Simple lexer to just parse the method definition and get all used names and types.
		/// Method code itself is parsed in are more complex (BNF), complete and slow way.
		/// </summary>
		private static string GetName(string firstLine)
		{
			var name = firstLine.SplitWordsAndPunctuation()[0];
			return name.IsKeyword() && !name.IsKeywordFunction()
				? throw new MethodNameCantBeKeyword(name)
				: name;
		}

		public class MethodNameCantBeKeyword : Exception
		{
			public MethodNameCantBeKeyword(string methodName) : base(methodName) { }
		}

		private void ParseDefinition(string rest)
		{
			var returnsIndex = rest.IndexOf(" " + Keyword.Returns + " ", StringComparison.Ordinal);
			if (returnsIndex >= 0)
			{
				ReturnType = Type.GetType(rest.Substring(returnsIndex + Keyword.Returns.Length + 2));
				rest = rest.Substring(0, returnsIndex);
			}
			if (string.IsNullOrEmpty(rest))
				return;
			if (rest == "()")
				throw new EmptyParametersMustBeRemoved();
			if (!rest.StartsWith('(') || !rest.EndsWith(')'))
				throw new InvalidSyntax(rest);
			ParseParameters(rest.Substring(1, rest.Length - 2));
		}

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
	}
}