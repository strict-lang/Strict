using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Strict.Language
{
	/// <summary>
	/// .strict files contain a type or trait and must be in the correct namespace folder.
	/// Strict code only contains optionally implement, then has*, then methods*. No empty lines.
	/// There is no typical lexing/scoping/token splitting needed as Strict syntax is very strict.
	/// </summary>
	public class Type : Context
	{
		public Type(Package package, string name, string code) : base(package, name)
		{
			if (package.FindDirectType(name) != null)
				throw new TypeAlreadyExistsInPackage(name, package);
			if (string.IsNullOrEmpty(code) && name != Base.None && name != Base.Boolean)
				throw new NoCodeGiven(name);
			Package = package;
			Package.Add(this);
			lines = code.SplitLines();
			if (name != Base.None && name != Base.Boolean)
				Parse();
		}

		public class TypeAlreadyExistsInPackage : Exception
		{
			public TypeAlreadyExistsInPackage(string name, Package package) : base(
				name + " in package: " + package) { }
		}

		public class NoCodeGiven : Exception
		{
			public NoCodeGiven(string name) : base(name) { }
		}

		public Package Package { get; }
		private readonly string[] lines;

		private void Parse()
		{
			try
			{
				for (lineNumber = 0; lineNumber < lines.Length; lineNumber++)
					ParseLine(lines[lineNumber]);
				if (methods.Count == 0)
					throw new NoMethodsFound(lineNumber, Name);
			}
			catch (Exception ex)
			{
				throw new ParsingFailed(ex, FilePath);
			}
		}

		public string FilePath => Path.Combine(Package.LocalPath, Name) + Extension;

		private int lineNumber;

		private void ParseLine(string line)
		{
			var words = ParseWords(line);
			if (lineNumber == 0 && words[0] == nameof(Implement).ToLower())
				implements.Add(new Implement(Package.GetType(words[1])));
			else if (words[0] == Keyword.Has)
				members.Add(new Member(this, line.Substring(Keyword.Has.Length + 1)));
			else if (words[0] == nameof(Method).ToLower())
				methods.Add(new Method(this, line, GetAllMethodLines()));
			else
				throw new InvalidLine(line, lineNumber, words.First());
		}

		private string[] ParseWords(string line)
		{
			if (line.Length != line.Trim().Length)
				throw new ExtraWhitespacesFound(line, lineNumber, line);
			if (line.Length == 0)
				throw new EmptyLine(lineNumber, Name);
			var words = line.SplitWords();
			if (words.Length == 1)
				throw new LineWithJustOneWord(line, lineNumber, words[0]);
			return words;
		}

		public class ExtraWhitespacesFound : LineException
		{
			public ExtraWhitespacesFound(string text, int line, string method) : base(text, line, method) { }
		}

		public abstract class LineException : Exception
		{
			protected LineException(string message, int line, string method) : base(message)
			{
				Number = line;
				Method = method;
			}

			public int Number { get; }
			public string Method { get; }
		}

		public class EmptyLine : LineException
		{
			public EmptyLine(int line, string method) : base("", line, method) { }
		}

		public class LineWithJustOneWord : LineException
		{
			public LineWithJustOneWord(string text, int line, string method) : base(text, line,
				method) { }
		}

		public class InvalidLine : LineException
		{
			public InvalidLine(string text, int line, string method) : base(text, line, method) { }
		}

		public class NoMethodsFound : LineException
		{
			public NoMethodsFound(int line, string method) : base("", line, method) { }
		}

		private IReadOnlyList<string> GetAllMethodLines()
		{
			if (IsTrait && lineNumber + 1 < lines.Length && lines[lineNumber + 1].StartsWith("\t"))
				throw new TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies();
			var methodLines = new List<string>();
			while (++lineNumber < lines.Length && lines[lineNumber].StartsWith("\t"))
				methodLines.Add(lines[lineNumber]);
			return methodLines;
		}

		public class TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies : Exception { }

		public class ParsingFailed : Exception
		{
			public ParsingFailed(Exception inner, string filePath) : base(
				inner is LineException line
					? "\n   at " + line.Method + " in " + filePath + ":line " + line.Number
					: filePath, inner) { }
		}

		public IReadOnlyList<Implement> Implements => implements;
		private readonly List<Implement> implements = new List<Implement>();
		public IReadOnlyList<Member> Members => members;
		private readonly List<Member> members = new List<Member>();
		public IReadOnlyList<Method> Methods => methods;
		private readonly List<Method> methods = new List<Method>();
		public bool IsTrait => Implements.Count == 0 && Members.Count == 0;

		public override string ToString() => base.ToString() + Implements.InBrackets();

		public override Type? FindType(string name, Package? searchingFromPackage = null,
			Type? searchingFromType = null) =>
			name == Name || name.Contains(".") && name == base.ToString()
				? this
				: Package.FindType(name, searchingFromPackage ?? Package, searchingFromType ?? this);

		//ncrunch: no coverage start, still needs to be tested
		public static Type FromFile(Package package, string filePath)
		{
			if (!filePath.EndsWith(Extension))
				throw new FileExtensionMustBeStrict();
			var directory = Path.GetDirectoryName(filePath)!;
			var paths = directory.Split(Path.DirectorySeparatorChar);
			if (package.Name != paths.Last())
				throw new FilePathMustMatchPackageName(package.Name, directory);
			if (!string.IsNullOrEmpty(package.Parent.Name) &&
				(paths!.Length < 2 || package.Parent.Name != paths[^2]))
				throw new FilePathMustMatchPackageName(package.Parent.Name, directory);
			return new Type(package, Path.GetFileNameWithoutExtension(filePath),
				File.ReadAllText(filePath));
		}

		public const string Extension = ".strict";

		public class FileExtensionMustBeStrict : Exception { }

		public class FilePathMustMatchPackageName : Exception
		{
			public FilePathMustMatchPackageName(string filePath, string packageName) : base(
				filePath + " must be in package folder " + packageName) { }
		}
	}
}