using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Strict.Language
{
	//TODO: get all the Strict.Parsing.TypeFileParser over here!
	/// <summary>
	/// Code files in strict must be in the correct namespace folder and end with .strict.
	/// Strict code only contains optionally implement, then has*, then methods*. No empty lines.
	/// There is no typical lexing/scoping/token splitting needed as Strict syntax is very strict.
	/// </summary>
	public class TypeParser //TODO: should just be type, CodeFile is just at parsing time, we don't really care about that anymore, we can always reconstruct the source code at any time we need it (just .ToString())
	{
		public TypeParser(Context context) => Context = context;
		public Context Context { get; }

		public Type ParseFile(string filePath)
		{
			//TODO: check context if this filePath is correct for the namespace, etc. we are in atm!
			return ParseLines(Path.GetFileNameWithoutExtension(filePath), File.ReadAllLines(filePath));
		}

		private Type ParseLines(string name, string[] lines)
		{
			//TODO: shouldn't this be at class level
			Implement? implement = null;
			List<Member> has = new List<Member>();
			List<Method> methods = new List<Method>();
			for (var lineNumber = 0; lineNumber < lines.Length; lineNumber++)
			{
				//TODO: expected tabs, cut them off
				var line = lines[lineNumber];
				if (line.Length != line.Trim().Length)
					throw new ExtraWhitespacesFound(line, lineNumber);
				if (line.Length == 0)
					throw new EmptyLine();
				var words = line.SplitWords();
				if (words.Length == 1)
					throw new LineWithJustOneWord(line, lineNumber);
				if (lineNumber == 0 && words[0] == nameof(Implement).ToLower())
				{
					implement = new Implement(new Trait(words[1], null));
					continue;
				}
				var isHasLine = words[0] == Keyword.Has;
				if (!isHasLine && implement == null && has.Count == 0)
					throw new MustStartWithImplementOrHas();
				var isMethodLine = words[0] == nameof(Method).ToLower();
				if (isHasLine)
					has.Add(new Member(words[1], Context.FindType(words.Last())));
				else if (isMethodLine)
				{
					methods.Add(new Method(words[1], new Parameter[0], Context.FindType(words.Last()))); //TODO: actual method parser
					lineNumber++;//dummy
				}
				else
					throw new InvalidSyntax(line, lineNumber);
			}//TODO: method to long, split up, maybe into parser class
			if (methods.Count == 0)
				throw new NoMethodsFound();
			return new Type(name, implement, has, methods);
		}

		public class ExtraWhitespacesFound : Exception
		{
			public ExtraWhitespacesFound(string line, int lineNumber) : base(
				line + " (" + lineNumber + ")") { }
		}

		public class EmptyLine : Exception{}

		public class LineWithJustOneWord : Exception
		{
			public LineWithJustOneWord(string line, int lineNumber) : base(
				line + " (" + lineNumber + ")") { }
		}
		
		public class MustStartWithImplementOrHas : Exception { }

		public class InvalidSyntax : Exception
		{
			public InvalidSyntax(string line, in int lineNumber) : base(
				line + " (" + lineNumber + ")") { }
		}

		public class NoMethodsFound : Exception{}

		public Type ParseCode(string name, string code) => ParseLines(name, code.SplitLines());
	}
}