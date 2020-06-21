﻿using System;
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
		public Type(Package package, string name, string code) : base(name)
		{
			if (package.FindType(name) != null)
				throw new TypeAlreadyExistsInPackage();
			Package = package;
			lines = code.SplitLines();
			Parse();
		}

		public class TypeAlreadyExistsInPackage : Exception { }

		public Package Package { get; }
		private readonly string[] lines;

		private void Parse()
		{
			for (lineNumber = 0; lineNumber < lines.Length; lineNumber++)
				ParseLine(lines[lineNumber]);
			if (methods.Count == 0)
				throw new NoMethodsFound();
		}
		
		private int lineNumber;
		
		private void ParseLine(string line)
		{
			var words = ParseWords(line);
			if (lineNumber == 0 && words[0] == nameof(Implement).ToLower())
				implements.Add(new Implement(Package.GetType(words[1])));
			else if (words[0] == Keyword.Has)
				members.Add(new Member(words[1], Package.GetType(words.Last())));
			else if (words[0] == nameof(Method).ToLower())
				methods.Add(new Method(this, line, GetAllMethodLines()));
			else
				throw new InvalidSyntax(line, lineNumber);
		}

		private string[] ParseWords(string line)
		{
			if (line.Length != line.Trim().Length)
				throw new ExtraWhitespacesFound(line, lineNumber);
			if (line.Length == 0)
				throw new EmptyLine();
			var words = line.SplitWords();
			if (words.Length == 1)
				throw new LineWithJustOneWord(line, lineNumber);
			return words;
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
		
		
		public class InvalidSyntax : Exception
		{
			public InvalidSyntax(string line, in int lineNumber) : base(
				line + " (" + lineNumber + ")") { }
		}

		public class NoMethodsFound : Exception{}

		private IReadOnlyList<string> GetAllMethodLines()
		{
			if (IsTrait && lineNumber+1 < lines.Length && lines[lineNumber+1].StartsWith("\t"))
				throw new TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies();
			var methodLines = new List<string>();
			while (++lineNumber < lines.Length && lines[lineNumber].StartsWith("\t"))
				methodLines.Add(lines[lineNumber]);
			return methodLines;
		}
		
		public class TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies : Exception { }
		public IReadOnlyList<Implement> Implements => implements;
		private readonly List<Implement> implements = new List<Implement>();
		public IReadOnlyList<Member> Members => members;
		private readonly List<Member> members = new List<Member>();
		public IReadOnlyList<Method> Methods => methods;
		private readonly List<Method> methods = new List<Method>();
		public bool IsTrait => Implements.Count == 0 && Members.Count == 0;
		
		public static Type FromFile(Package package, string filePath)
		{
			//TODO: also check context if this filePath is correct for the namespace, etc. we are in atm!
			return new Type(package, Path.GetFileNameWithoutExtension(filePath),
				File.ReadAllText(filePath));
		}

		public override string ToString() => Name + Implements.ToWordString();
	}
}