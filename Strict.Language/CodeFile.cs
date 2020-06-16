using System;
using System.Collections.Generic;
using System.IO;

namespace Strict.Language
{
	//TODO: get all the Strict.Parsing.TypeFileParser over here!
	/// <summary>
	/// Code files in strict must be in the correct namespace folder and end with .strict.
	/// Strict code only contains optionally implement, then has*, then methods*. No empty lines.
	/// </summary>
	public class CodeFile
	{
		/*TODO
		public CodeFile(string filePath)
		{
			//TODO: extract name
			Parse(File.ReadAllLines(filePath));
		}*/

		private CodeFile Parse(string[] lines)
		{
			for (var lineNumber = 0; lineNumber < lines.Length; lineNumber++)
			{
				//TODO: expected tabs, cut them off
				var line = lines[lineNumber];
				if (line.Length != line.Trim().Length)
					throw new ExtraWhitespacesFound(line, lineNumber);
				if (line.Length == 0)
					throw new EmptyLine();
				var words = line.Split(' ');
				if (words.Length == 1)
					throw new LineWithJustOneWord(line, lineNumber);
				if (lineNumber == 0 && words[0] == nameof(Implement).ToLower())
				{
					Implement = new Keyword(words[1]);
					continue;
				}
				var isHasLine = words[0] == nameof(Has).ToLower();
				if (!isHasLine && Implement == null && has.Count == 0)
					throw new MustStartWithImplementOrHas();
				var isMethodLine = words[0] == nameof(Method).ToLower();
				if (isHasLine)
					has.Add(new Keyword(words[1]));//TODO: type
				else if (isMethodLine)
				{
					methods.Add(new Method(words[1])); //TODO: actual method parser
					lineNumber++;//dummy
				}
				else
					throw new InvalidSyntax(line, lineNumber);
			}//TODO: method to long, split up, maybe into parser class
			if (methods.Count == 0)
				throw new NoMethodsFound();
			return this;
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

		public static CodeFile FromCode(string code)//TODO: add name
			=>
				new CodeFile().Parse(SplitLines(code));

		private static string[] SplitLines(string text)
			=> text.Split(new[] { Environment.NewLine, "\n" }, StringSplitOptions.None);

		private CodeFile() { }
		public Keyword? Implement { get; private set; }
		public IReadOnlyList<Keyword> Has => has;
		public readonly List<Keyword> has = new List<Keyword>();
		public IReadOnlyList<Keyword> Methods => methods;
		public readonly List<Keyword> methods = new List<Keyword>();
	}

	/*TODO?
	public class Implement : Keyword
	{

	}
	*/
	public class Method : Keyword
	{
		public Method(string name) : base(name) { }
	}

	public class Keyword
	{
		public Keyword(string name) => Name = name;
		public string Name { get; }
	}
}