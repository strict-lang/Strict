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
		public Method(Type type, string firstLine, IReadOnlyList<string> lines) : this(type, new LineLexer(firstLine), lines){}

		public Method(Type type, LineLexer definition, IReadOnlyList<string> lines) : base(GetName(definition))
		{
			this.type = type;
			while (definition.HasNext)
			{
				var parameterName = definition.Next();
				if (parameterName == Keyword.Returns)
					break;
				parameters.Add(new Parameter(parameterName, type.GetType(parameterName)));
			}
			ReturnType = Name == Keyword.From
				? type
				: type.GetType(definition.HasNext
					? definition.Next()
					: Base.None);
			this.lines = lines;
			//TODO: parse lazily: var lines = code.SplitLines();
		}

		private static string GetName(LineLexer definition) =>
			definition.Next() switch
			{
				Keyword.Method => definition.Next(),
				Keyword.From => Keyword.From,
				_ => throw new InvalidSyntax()
			};

		private readonly Type type;

		public class InvalidSyntax : Exception { }

		public IReadOnlyList<Parameter> Parameters => parameters;
		private readonly List<Parameter> parameters = new List<Parameter>();
		public Type ReturnType { get; }
		private IReadOnlyList<string> lines;
	}
}