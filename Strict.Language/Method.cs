using System;
using System.Collections.Generic;

namespace Strict.Language
{
	/// <summary>
	/// Methods are parsed lazily, which speeds up type and package parsing enormously and
	/// also provides us with all methods in a type usable in any other method if needed.
	/// </summary>
	public class Method : Context
	{
		public Method(Type type, string firstLine, IReadOnlyList<string> lines) : base(type,
			GetName(firstLine))
		{
			ReturnType = Name == Keyword.From ? type : type.GetType(Base.None);
			ParseDefinition(Name == Keyword.From
				? firstLine.Substring(Keyword.From.Length)
				: firstLine.Substring(Keyword.Method.Length + 1 + Name.Length));
			this.lines = lines;
		}

		/// <summary>
		/// Simple lexer to just parse the method definition and get all used names and types.
		/// Method code itself is parsed in are more complex (BNF), complete and slow way.
		/// </summary>
		private static string GetName(string firstLine)
		{
			if (firstLine.StartsWith(Keyword.From + "("))
				return Keyword.From;
			if (!firstLine.StartsWith(Keyword.Method + " "))
				throw new InvalidMethodDefinition(firstLine);
			var name = firstLine.SplitWordsAndPunctuation()[1];
			if (Keyword.IsKeyword(name))
				throw new MethodNameCantBeKeyword(name);
			return name;
		}

		public class MethodNameCantBeKeyword : Exception
		{
			public MethodNameCantBeKeyword(string methodName) : base(methodName) { }
		}

		public class InvalidMethodDefinition : Exception
		{
			public InvalidMethodDefinition(string line) : base(line) { }
		}

		private void ParseDefinition(string rest)
		{
			var returnsIndex =
				rest.IndexOf(" " + Keyword.Returns + " ", StringComparison.InvariantCulture);
			if (returnsIndex >= 0)
			{
				ReturnType = Type.GetType(rest.Substring(returnsIndex + Keyword.Returns.Length + 2));
				rest = rest.Substring(0, returnsIndex);
			}
			if (string.IsNullOrEmpty(rest))
				return;
			if (rest.StartsWith("(") && rest.EndsWith(")"))
				ParseParameters(rest.Substring(1, rest.Length - 2));
			else
				throw new InvalidSyntax(rest);
		}

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
		// ReSharper disable once NotAccessedField.Local
		private IReadOnlyList<string> lines;

		public override Type? FindType(string name, Type? searchingFromType = null) =>
			Type.FindType(name, searchingFromType);
	}
}