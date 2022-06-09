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
	/// Call Parse instead. This just sets the type name in the specified package.
	/// </summary>
	public Type(Package package, string filePath, ExpressionParser expressionParser) : base(package,
		Path.GetFileNameWithoutExtension(filePath))
	{
		if (package.FindDirectType(Name) != null)
			throw new TypeAlreadyExistsInPackage(Name, package);
		Package = package;
		Package.Add(this);
		FilePath = Name == filePath
			? Path.Combine(Package.FolderPath, Name) + Extension
			: filePath;
		this.expressionParser = expressionParser;
	}

	public sealed class TypeAlreadyExistsInPackage : Exception
	{
		public TypeAlreadyExistsInPackage(string name, Package package) : base(
			name + " in package: " + package) { }
	}

	public Package Package { get; }
	public string FilePath { get; }
	private readonly ExpressionParser expressionParser;
	public Type Parse(string setLines) => Parse(setLines.SplitLines());

	public Type Parse(string[] setLines)
	{
		lines = setLines;
		for (lineNumber = 0; lineNumber < lines.Length; lineNumber++)
			TryParseLine(lines[lineNumber]);
		if (methods.Count == 0)
			throw new NoMethodsFound(this, lineNumber);
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
			throw new MustImplementAllTraitMethods(this, nonImplementedTraitMethods);
	}

	private string[] lines = Array.Empty<string>();
	private int lineNumber;

	private void TryParseLine(string line)
	{
		try
		{
			ParseLine(line);
		}
		catch (ParsingFailed)
		{
			throw;
		}
		catch (Exception ex)
		{
			throw new ParsingFailed(this, lineNumber, line, ex);
		}
	}

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
			throw new ExtraWhitespacesFoundAtBeginningOfLine(this, lineNumber, line);
		if (line.Length != line.TrimEnd().Length)
			throw new ExtraWhitespacesFoundAtEndOfLine(this, lineNumber, line);
		if (line.Length == 0)
			throw new EmptyLineIsNotAllowed(this, lineNumber);
		return line.SplitWords();
	}

	public sealed class ExtraWhitespacesFoundAtBeginningOfLine : ParsingFailed
	{
		public ExtraWhitespacesFoundAtBeginningOfLine(Type type, int lineNumber, string message,
			string method = "") : base(type, lineNumber, message, method) { }
	}

	public sealed class ExtraWhitespacesFoundAtEndOfLine : ParsingFailed
	{
		public ExtraWhitespacesFoundAtEndOfLine(Type type, int lineNumber, string message,
			string method = "") : base(type, lineNumber, message, method) { }
	}

	public sealed class EmptyLineIsNotAllowed : ParsingFailed
	{
		public EmptyLineIsNotAllowed(Type type, int lineNumber) : base(type, lineNumber) { }
	}

	public sealed class NoMethodsFound : ParsingFailed
	{
		public NoMethodsFound(Type type, int lineNumber) : base(type, lineNumber,
			"Each type must have at least one method, otherwise it is useless") { }
	}

	public sealed class MustImplementAllTraitMethods : ParsingFailed
	{
		public MustImplementAllTraitMethods(Type type, IEnumerable<Method> missingTraitMethods) :
			base(type, type.lineNumber, "Missing methods: " + string.Join(", ", missingTraitMethods)) { }
	}

	private string[] GetAllMethodLines(string definitionLine)
	{
		var methodLines = new List<string> { definitionLine };
		if (IsTrait && IsNextLineValidMethodBody())
			throw new TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies(this);
		if (!IsTrait && !IsNextLineValidMethodBody())
			throw new MethodMustBeImplementedInNonTraitType(this, definitionLine);
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
			throw new ExtraWhitespacesFoundAtBeginningOfLine(this, lineNumber, line);
		return false;
	}

	public sealed class TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies : ParsingFailed
	{
		public TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies(Type type) : base(type, 0) { }
	}

	// ReSharper disable once HollowTypeName
	public sealed class MethodMustBeImplementedInNonTraitType : ParsingFailed
	{
		public MethodMustBeImplementedInNonTraitType(Type type, string definitionLine) : base(type,
			type.lineNumber, definitionLine) { }
	}

	public IReadOnlyList<Type> Implements => implements;
	private readonly List<Type> implements = new();
	public IReadOnlyList<Package> Imports => imports;
	private readonly List<Package> imports = new();
	public IReadOnlyList<Member> Members => members;
	private readonly List<Member> members = new();
	public IReadOnlyList<Method> Methods => methods;
	protected readonly List<Method> methods = new();
	public bool IsTrait => Implements.Count == 0 && Members.Count == 0 && Name != Base.Number;
	public override string ToString() => base.ToString() + Implements.ToBrackets();

	//TODO: To save the type and the method into the text file, may be use a ToSourceFile() method

	public override Type? FindType(string name, Context? searchingFrom = null) =>
		name == Name || name.Contains('.') && name == base.ToString()
			? this
			: Package.FindType(name, searchingFrom ?? this);

	public async Task ParseFile(string filePath)
	{
		if (!filePath.EndsWith(Extension, StringComparison.Ordinal))
			throw new FileExtensionMustBeStrict(filePath);
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

	public sealed class FileExtensionMustBeStrict : Exception
	{
		public FileExtensionMustBeStrict(string filePath) : base(filePath) { }
	}

	public sealed class FilePathMustMatchPackageName : Exception
	{
		public FilePathMustMatchPackageName(string filePath, string packageName) : base(filePath +
			" must be in package folder " + packageName) { }
	}
}