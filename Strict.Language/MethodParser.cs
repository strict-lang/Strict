using System;
using System.Collections.Generic;

namespace Strict.Language
{
	public class MethodParser
	{
		public MethodParser(Context context) => this.context = context;
		private readonly Context context;

		public Method Parse(string code)
		{
			var lines = code.SplitLines();
			var definition = new LineLexer(lines[0]);
			var name = definition.Next() switch
			{
				Keyword.Method => definition.Next(),
				Keyword.From => Keyword.From,
				_ => throw new InvalidSyntax()
			};
			var parameters = new List<Parameter>();
			while (definition.HasNext)
			{
				var parameterName = definition.Next();
				if (parameterName == Keyword.Returns)
					break;
				parameters.Add(new Parameter(parameterName, context.FindType(parameterName)));
			}
			var returnType = definition.HasNext
				? context.FindType(definition.Next())
				: context.ParentType;
			return new Method(name, parameters, returnType);
		}

		public class InvalidSyntax : Exception { }
	}
}