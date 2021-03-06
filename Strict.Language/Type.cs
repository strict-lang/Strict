﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace Strict.Language
{
	/// <summary>
	/// .strict files contain a type or trait and must be in the correct namespace folder.
	/// Strict code only contains optionally implement, then has*, then methods*. No empty lines.
	/// There is no typical lexing/scoping/token splitting needed as Strict syntax is very strict.
	/// </summary>
	public class Type : Context
	{
		/// <summary>
		/// Call ParserCode or ParseFile instead. This just sets the type name in the specified package.
		/// </summary>
		public Type(Package package, string name, ExpressionParser expressionParser) : base(package,
			name)
		{
			if (package.FindDirectType(name) != null)
				throw new TypeAlreadyExistsInPackage(name, package);
			Package = package;
			Package.Add(this);
			this.expressionParser = expressionParser;
		}

		public class TypeAlreadyExistsInPackage : Exception
		{
			public TypeAlreadyExistsInPackage(string name, Package package) : base(
				name + " in package: " + package) { }
		}

		public Package Package { get; }
		private readonly ExpressionParser expressionParser;
		public Type Parse(string code) => Parse(code.SplitLines());

		public Type Parse(string[] setLines)
		{
			try
			{
				lines = setLines;
				for (lineNumber = 0; lineNumber < lines.Length; lineNumber++)
					ParseLine(lines[lineNumber]);
				if (methods.Count == 0)
					throw new NoMethodsFound(lineNumber, Name);
				return this;
			}
			catch (ParsingFailedInLine line)
			{
				throw new ParsingFailed(line, FilePath);
			}
			catch (Exception ex)
			{
				throw new ParsingFailed(ex, (lineNumber < lines.Length
					? lines[lineNumber].SplitWords().First()
					: "") + " in " + FilePath + ":line " + lineNumber);
			}
		}

		private string[] lines = new string[0];
		private int lineNumber;
		public string FilePath => Path.Combine(Package.FolderPath, Name) + Extension;

		private void ParseLine(string line)
		{
			var words = ParseWords(line);
			if (words[0] == Import)
				imports.Add(ParseImport(words));
			else if (words[0] == Implement)
				implements.Add(ParseImplement(words));
			else if (words[0] == Has)
				members.Add(ParseMember(line));
			else
				methods.Add(new Method(this, expressionParser, GetAllMethodLines(line)));
		}

		private Package ParseImport(string[] words)
		{
			if (implements.Count > 0 || members.Count > 0 || methods.Count > 0)
				throw new ImportMustBeFirst(words[1]);
			var import = Package.Find(words[1]);
			if (import == null)
				throw new PackageNotFound(words[1]);
			return import;
		}

		public class ImportMustBeFirst : Exception
		{
			public ImportMustBeFirst(string package) : base(package) { }
		}

		public class PackageNotFound : Exception
		{
			public PackageNotFound(string package) : base(package) { }
		}

		private Type ParseImplement(string[] words)
		{
			if (members.Count > 0 || methods.Count > 0)
				throw new ImplementMustComeBeforeMembersAndMethods(words[1]);
			return Package.GetType(words[1]);
		}

		public class ImplementMustComeBeforeMembersAndMethods : Exception
		{
			public ImplementMustComeBeforeMembersAndMethods(string type) : base(type) { }
		}

		private Member ParseMember(string line)
		{
			if (methods.Count > 0)
				throw new MembersMustComeBeforeMethods(line);
			return new(this, line.Substring(Has.Length + 1), null!);
		}

		public class MembersMustComeBeforeMethods : Exception
		{
			public MembersMustComeBeforeMethods(string line) : base(line) { }
		}

		public const string Implement = "implement";
		public const string Import = "import";
		public const string Has = "has";

		private string[] ParseWords(string line)
		{
			if (line.Length != line.TrimStart().Length)
				throw new ExtraWhitespacesFoundAtBeginningOfLine(line, lineNumber, line);
			if (line.Length != line.TrimEnd().Length)
				throw new ExtraWhitespacesFoundAtEndOfLine(line, lineNumber, line);
			if (line.Length == 0)
				throw new EmptyLine(lineNumber, Name);
			return line.SplitWords();
		}

		public class ExtraWhitespacesFoundAtBeginningOfLine : ParsingFailedInLine
		{
			public ExtraWhitespacesFoundAtBeginningOfLine(string text, int line, string method) : base(
				text, line, method) { }
		}

		public class ExtraWhitespacesFoundAtEndOfLine : ParsingFailedInLine
		{
			public ExtraWhitespacesFoundAtEndOfLine(string text, int line, string method) : base(text,
				line, method) { }
		}

		public abstract class ParsingFailedInLine : Exception
		{
			protected ParsingFailedInLine(string message, int line, string method) : base(message)
			{
				Number = line;
				Method = method;
			}

			public int Number { get; }
			public string Method { get; }
		}

		public class EmptyLine : ParsingFailedInLine
		{
			public EmptyLine(int line, string method) : base("", line, method) { }
		}

		public class NoMethodsFound : ParsingFailedInLine
		{
			public NoMethodsFound(int line, string method) : base(
				"Each type must have at least one method, otherwise it is useless", line, method) { }
		}

		private string[] GetAllMethodLines(string definitionLine)
		{
			if (IsTrait && lineNumber + 1 < lines.Length && lines[lineNumber + 1].StartsWith('\t'))
				throw new TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies();
			var methodLines = new List<string> { definitionLine };
			while (lineNumber + 1 < lines.Length && lines[lineNumber + 1].StartsWith('\t'))
				methodLines.Add(lines[++lineNumber]);
			return methodLines.ToArray();
		}

		public class TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies : Exception { }

		public class ParsingFailed : Exception
		{
			public ParsingFailed(ParsingFailedInLine line, string filePath) : base(
				"\n   at " + line.Method + " in " + filePath + ":line " + line.Number, line) { }

			public ParsingFailed(Exception inner, string fallbackWordAndLineNumber) : base(
				"\n   at " + fallbackWordAndLineNumber, inner) { }
		}

		public IReadOnlyList<Type> Implements => implements;
		private readonly List<Type> implements = new();
		public IReadOnlyList<Package> Imports => imports;
		private readonly List<Package> imports = new();
		public IReadOnlyList<Member> Members => members;
		private readonly List<Member> members = new();
		public IReadOnlyList<Method> Methods => methods;
		private readonly List<Method> methods = new();
		public bool IsTrait => Implements.Count == 0 && Members.Count == 0 && Name != Base.Number;
		public override string ToString() => base.ToString() + Implements.ToBracketsString();

		public override Type? FindType(string name, Context? searchingFrom = null) =>
			name == Name || name.Contains(".") && name == base.ToString()
				? this
				: Package.FindType(name, searchingFrom ?? this);

		public async Task ParseFile(string filePath)
		{
			if (!filePath.EndsWith(Extension))
				throw new FileExtensionMustBeStrict();
			var directory = Path.GetDirectoryName(filePath)!;
			var paths = directory.Split(Path.DirectorySeparatorChar);
			if (Package.Name != paths.Last())
				throw new FilePathMustMatchPackageName(Package.Name, directory);
			if (!string.IsNullOrEmpty(Package.Parent.Name) &&
				(paths!.Length < 2 || Package.Parent.Name != paths[^2]))
				throw new FilePathMustMatchPackageName(Package.Parent.Name, directory);
			Parse(await File.ReadAllLinesAsync(filePath));
		}

		public const string Extension = ".strict";

		public class FileExtensionMustBeStrict : Exception { }

		public class FilePathMustMatchPackageName : Exception
		{
			public FilePathMustMatchPackageName(string filePath, string packageName) : base(filePath +
				" must be in package folder " + packageName) { }
		}
	}
}