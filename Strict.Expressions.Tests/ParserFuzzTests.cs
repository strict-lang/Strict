using System.Diagnostics;
using System.Text.RegularExpressions;

namespace Strict.Expressions.Tests;

/// <summary>
/// Mutates every .strict file of the repository and parses each mutation in memory: the parser may
/// only report ParsingFailed errors, never crash with a .NET exception (not even wrapped) or hang.
/// </summary>
public sealed class ParserFuzzTests
{
	[Test]
	[Category("Slow")]
	public async Task MutatedStrictFilesOnlyFailWithParsingFailed()
	{
		var repositories = new Repositories(Parser);
		var crashes = new Dictionary<string, List<string>>();
		foreach (var file in StrictFiles())
		{
			var random = new Random(GetSeed(file));
			var package = await repositories.LoadStrictPackage(GetPackageName(file));
			var privatePackage = CreatePrivateTwin(package);
			var lines = TypeLines.FromFile(file);
			for (var count = 0; count < MutationsPerFile; count++)
			{
				var mutatedLines = Mutate(lines, random, out var mutation);
				var parse = Task.Run(() =>
					Parse(privatePackage, Path.GetFileNameWithoutExtension(file), mutatedLines));
				if (!parse.Wait(ParseTimeout))
					Assert.Fail("Parsing hangs: " + file + " " + mutation);
				if (parse.Result is not { } crash)
					continue;
				var crashClass = GetCrashClass(crash);
				if (!crashes.TryGetValue(crashClass, out var examples))
					crashes[crashClass] = examples = [];
				examples.Add(file + " " + mutation + "\n" + crash);
			}
		}
		Assert.That(crashes, Is.Empty, string.Join("\n\n", crashes.OrderByDescending(pair =>
			pair.Value.Count).Select(pair => pair.Value.Count + "x " + pair.Key + "\n" + pair.Value[0])));
	}

	private static readonly MethodExpressionParser Parser = new();
	private const int MutationsPerFile = 40;
	private static readonly TimeSpan ParseTimeout = TimeSpan.FromSeconds(5);
	private static readonly string Root =
		Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict));

	/// <summary>
	/// All .strict files of the root and every package folder, C# project folders are skipped.
	/// </summary>
	private static IEnumerable<string> StrictFiles() =>
		Directory.EnumerateFiles(Root, "*" + Type.Extension).
			Concat(Directory.EnumerateDirectories(Root).Where(IsPackageFolder).SelectMany(folder =>
				Directory.EnumerateFiles(folder, "*" + Type.Extension, SearchOption.AllDirectories))).
			Where(file => !IsBuildOutput(file)).Order(StringComparer.Ordinal);

	/// <summary>
	/// Seeded per file from its repository path (string.GetHashCode is randomized per process), so
	/// adding or editing other .strict files never changes the mutations of this file.
	/// </summary>
	private static int GetSeed(string file) =>
		Path.GetRelativePath(Root, file).Replace('\\', '/').Aggregate(7,
			(hash, character) => hash * 31 + character);

	private static bool IsPackageFolder(string folder) =>
		!Path.GetFileName(folder).StartsWith('.') &&
		!Path.GetFileName(folder).StartsWith(nameof(Strict), StringComparison.Ordinal);

	private static bool IsBuildOutput(string file) =>
		file.Contains(Path.DirectorySeparatorChar + "bin" + Path.DirectorySeparatorChar) ||
		file.Contains(Path.DirectorySeparatorChar + "obj" + Path.DirectorySeparatorChar);

	private static string GetPackageName(string file) =>
		Path.GetRelativePath(Root, Path.GetDirectoryName(file)!).
			Replace('\\', Context.ParentSeparator) is var folder && folder == "."
			? nameof(Strict)
			: nameof(Strict) + Context.ParentSeparator + folder;

	/// <summary>
	/// ponytail: a package named like an existing child is not registered in the parent (see
	/// DISABLE_DISPOSING in Package), so mutated types are invisible to all other tests. Upgrade
	/// path: an explicit private package constructor once Package disposing is enabled again.
	/// </summary>
	private static Package CreatePrivateTwin(Package package) =>
		new((Package)package.Parent, package.FolderPath)
		{
			automaticallyLoadedDependencyPackages =
				[package, .. package.automaticallyLoadedDependencyPackages]
		};

	private static string[] Mutate(string[] lines, Random random, out string mutation)
	{
		var index = random.Next(lines.Length);
		var line = lines[index];
		var position = random.Next(line.Length + 1);
		var kind = random.Next(MutationNames.Length);
		var mutated = kind switch
		{
			0 => [.. lines[..index], .. lines[(index + 1)..]],
			1 => [.. lines[..(index + 1)], .. lines[index..]],
			2 => index + 1 < lines.Length
				? [.. lines[..index], lines[index + 1], line, .. lines[(index + 2)..]]
				: lines,
			_ => [.. lines[..index], MutateLine(kind, line, position, random), .. lines[(index + 1)..]]
		};
		mutation = "line " + (index + 1) + " " + MutationNames[kind] + ": " +
			(index < mutated.Length
				? mutated[index]
				: "");
		return mutated;
	}

	private static readonly string[] MutationNames =
	[
		"deleted", "duplicated", "swapped with next", "tab removed", "tab added",
		"character deleted", "character inserted", "truncated", "word replaced by keyword"
	];

	private static string MutateLine(int kind, string line, int position, Random random) =>
		kind switch
		{
			3 => line.IndexOf('\t') is var tab and >= 0
				? line.Remove(tab, 1)
				: line,
			4 => "\t" + line,
			5 => position < line.Length
				? line.Remove(position, 1)
				: line,
			6 => line.Insert(position, InsertCharacters[random.Next(InsertCharacters.Length)].ToString()),
			7 => line[..position],
			_ => Words.Matches(line) is { Count: > 0 } words &&
				words[random.Next(words.Count)] is var word
					? line[..word.Index] + Keywords[random.Next(Keywords.Length)] +
					line[(word.Index + word.Length)..]
					: line
		};

	private const string InsertCharacters = "()\",=. ";
	private static readonly Regex Words = new("[A-Za-z]+", RegexOptions.Compiled);
	private static readonly string[] Keywords =
		[.. Keyword.GetAllKeywords, "is", "not", "to", "then", "in", "and", "or"];

	private static Exception? Parse(Package package, string typeName, string[] lines)
	{
		Type? type = null;
		try
		{
			type = new Type(package, new TypeLines(typeName, lines));
			type.ParseMembersAndMethods(Parser);
			type.ValidateMembersAndVariablesAreUsed();
			foreach (var method in type.Methods)
				if (!method.IsTrait)
					method.GetBodyAndParseIfNeeded(type.IsGeneric);
			return null;
		}
		catch (Exception exception)
		{
			return exception is not ParsingFailed || ContainsDotNetException(exception.InnerException)
				? exception
				: null;
		}
		finally
		{
			type?.Dispose();
		}
	}

	/// <summary>
	/// The parser wraps every exception into ParsingFailed, a .NET exception inside is still a crash.
	/// </summary>
	private static bool ContainsDotNetException(Exception? exception) =>
		exception != null && (exception.GetType().Namespace?.StartsWith(nameof(Strict),
			StringComparison.Ordinal) != true || ContainsDotNetException(exception.InnerException));

	private static string GetCrashClass(Exception exception)
	{
		while (exception.InnerException != null)
			exception = exception.InnerException;
		var thrownIn = new StackTrace(exception).GetFrames().Select(frame => frame.GetMethod()).
			FirstOrDefault(method => method?.DeclaringType?.Namespace?.StartsWith(nameof(Strict),
				StringComparison.Ordinal) == true);
		return exception.GetType().Name + " in " + thrownIn?.DeclaringType?.Name + "." + thrownIn?.Name;
	}
}
