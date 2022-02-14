using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace Strict.Language;

/// <summary>
/// .strict files contain a type or trait and must be in the correct namespace folder.
/// Strict code only contains optionally implement, then has*, then methods*. No empty lines.
/// There is no typical lexing/scoping/token splitting needed as Strict syntax is very strict.
/// </summary>
// ReSharper disable once HollowTypeName
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

	public sealed class TypeAlreadyExistsInPackage : Exception
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
			return TryParse(setLines);
		}
		catch (ParsingFailedInLine line)
		{
			throw new ParsingFailed(line, FilePath);
		}
		catch (Exception ex)
		{
			throw new ParsingFailed(ex, (lineNumber < lines.Length
				? lines[lineNumber].SplitWords().First()
				: "") + " in " + FilePath + ":line " + (lineNumber + 1));
		}
	}

	private Type TryParse(string[] setLines)
	{
		lines = setLines;
		for (lineNumber = 0; lineNumber < lines.Length; lineNumber++)
			ParseLine(lines[lineNumber]);
		if (methods.Count == 0)
			throw new NoMethodsFound(lineNumber, Name);
		foreach (var trait in implements)
			if (trait.IsTrait)
				CheckIfTraitIsImplemented(trait);
		return this;
	}

	private void CheckIfTraitIsImplemented(Type trait)
	{
		var nonImplementedTraitMethods = new List<Method>();
		foreach (var traitMethod in trait.Methods)
			if (traitMethod.Name != Method.From && methods.All(implementedMethod =>
				traitMethod.Name != implementedMethod.Name))
				nonImplementedTraitMethods.Add(traitMethod);
		if (nonImplementedTraitMethods.Count > 0)
			throw new MustImplementAllTraitMethods(nonImplementedTraitMethods);
	}

	private string[] lines = Array.Empty<string>();
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
			methods.Add(new Method(this, lineNumber, expressionParser, GetAllMethodLines(line)));
	}

	private Package ParseImport(IReadOnlyList<string> words)
	{
		if (implements.Count > 0 || members.Count > 0 || methods.Count > 0)
			throw new ImportMustBeFirst(words[1]);
		var import = Package.Find(words[1]);
		if (import == null)
			throw new PackageNotFound(words[1]);
		return import;
	}

	public sealed class ImportMustBeFirst : Exception
	{
		public ImportMustBeFirst(string package) : base(package) { }
	}

	public sealed class PackageNotFound : Exception
	{
		public PackageNotFound(string package) : base(package) { }
	}

	private Type ParseImplement(IReadOnlyList<string> words)
	{
		if (members.Count > 0 || methods.Count > 0)
			throw new ImplementMustComeBeforeMembersAndMethods(words[1]);
		if (words[1] == "Any")
			throw new ImplementAnyIsImplicitAndNotAllowed();
		return Package.GetType(words[1]);
	}

	public sealed class ImplementMustComeBeforeMembersAndMethods : Exception
	{
		public ImplementMustComeBeforeMembersAndMethods(string type) : base(type) { }
	}

	public sealed class ImplementAnyIsImplicitAndNotAllowed : Exception { }

	private Member ParseMember(string line)
	{
		if (methods.Count > 0)
			throw new MembersMustComeBeforeMethods(line);
		var nameAndExpression = line[(Has.Length + 1)..].Split(" = ");
		var expression = nameAndExpression.Length > 1
			? expressionParser.ParseAssignmentExpression(new Member(this, nameAndExpression[0], null!).Type,
				nameAndExpression[1], lineNumber)
			: null;
		return new Member(this, nameAndExpression[0], expression);
	}

	public sealed class MembersMustComeBeforeMethods : Exception
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
			throw new EmptyLineIsNotAllowed(lineNumber, Name);
		return line.SplitWords();
	}

	public sealed class ExtraWhitespacesFoundAtBeginningOfLine : ParsingFailedInLine
	{
		public ExtraWhitespacesFoundAtBeginningOfLine(string text, int line, string method) : base(
			text, line, method) { }
	}

	public sealed class ExtraWhitespacesFoundAtEndOfLine : ParsingFailedInLine
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

	public sealed class EmptyLineIsNotAllowed : ParsingFailedInLine
	{
		public EmptyLineIsNotAllowed(int line, string method) : base("", line, method) { }
	}

	public sealed class NoMethodsFound : ParsingFailedInLine
	{
		public NoMethodsFound(int line, string method) : base(
			"Each type must have at least one method, otherwise it is useless", line, method) { }
	}

	public sealed class MustImplementAllTraitMethods : Exception
	{
		public MustImplementAllTraitMethods(IEnumerable<Method> missingTraitMethods) : base(
			"Missing methods: " + string.Join(", ", missingTraitMethods)) { }
	}

	private string[] GetAllMethodLines(string definitionLine)
	{
		var methodLines = new List<string> { definitionLine };
		if (IsTrait && IsNextLineValidMethodBody())
			throw new TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies();
		if (!IsTrait && !IsNextLineValidMethodBody())
			throw new MethodMustBeImplementedInNonTraitType(definitionLine);
		while (IsNextLineValidMethodBody())
			methodLines.Add(lines[++lineNumber]);
		return methodLines.ToArray();
	}

	private bool IsNextLineValidMethodBody()
	{
		if (lineNumber + 1 >= lines.Length)
			return false;
		var line = lines[lineNumber + 1];
		if (line.StartsWith('\t'))
			return true;
		if (line.Length != line.TrimStart().Length)
			throw new ExtraWhitespacesFoundAtBeginningOfLine(line, lineNumber, line);
		return false;
	}

	public sealed class TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies : Exception { }

	// ReSharper disable once HollowTypeName
	public sealed class MethodMustBeImplementedInNonTraitType : Exception
	{
		public MethodMustBeImplementedInNonTraitType(string definitionLine) : base(definitionLine) { }
	}

	public sealed class ParsingFailed : Exception
	{
		public ParsingFailed(ParsingFailedInLine line, string filePath) : base(
			"\n   at " + line.Method + " in " + filePath + ":line " + (line.Number + 1), line) { }

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
	public override string ToString() => base.ToString() + Implements.ToBrackets();

	public override Type? FindType(string name, Context? searchingFrom = null) =>
		name == Name || name.Contains('.') && name == base.ToString()
			? this
			: Package.FindType(name, searchingFrom ?? this);

	public async Task ParseFile(string filePath)
	{
		if (!filePath.EndsWith(Extension, StringComparison.Ordinal))
			throw new FileExtensionMustBeStrict();
		var directory = Path.GetDirectoryName(filePath)!;
		var paths = directory.Split(Path.DirectorySeparatorChar);
		CheckForFilePathErrors(filePath, paths, directory);
		Parse(await File.ReadAllLinesAsync(filePath));
	}

	private void CheckForFilePathErrors(string filePath, IReadOnlyList<string> paths, string directory)
	{
		if (Package.Name != paths.Last())
			throw new FilePathMustMatchPackageName(Package.Name, directory);
		if (!string.IsNullOrEmpty(Package.Parent.Name) &&
			(paths.Count < 2 || Package.Parent.Name != paths[^2]))
			throw new FilePathMustMatchPackageName(Package.Parent.Name, directory);
		if (directory.EndsWith(@"\strict-lang\Strict", StringComparison.Ordinal))
			throw new StrictFolderIsNotAllowedForRootUseBaseSubFolder(filePath); //ncrunch: no coverage
	}

	//ncrunch: no coverage start, tests too flacky when creating and deleting wrong file
	public sealed class StrictFolderIsNotAllowedForRootUseBaseSubFolder : Exception
	{
		public StrictFolderIsNotAllowedForRootUseBaseSubFolder(string filePath) : base(filePath) { }
	} //ncrunch: no coverage end

	public const string Extension = ".strict";
	public sealed class FileExtensionMustBeStrict : Exception { }

	public sealed class FilePathMustMatchPackageName : Exception
	{
		public FilePathMustMatchPackageName(string filePath, string packageName) : base(filePath +
			" must be in package folder " + packageName) { }
	}
}