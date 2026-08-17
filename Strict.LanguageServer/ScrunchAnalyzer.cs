using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.LanguageServer;

public sealed class FileReport
{
	public required string Path { get; init; }
	public bool Cached { get; init; }
	public bool Ok { get; init; }
	public int TestsPassed { get; init; }
	public int TestsFailed { get; init; }
	public List<Problem> Problems { get; init; } = [];
}

public sealed class Problem
{
	public int Line { get; init; }
	public required string Kind { get; init; }
	public required string Message { get; init; }
}

public sealed class FolderReport
{
	public required string Path { get; init; }
	public bool Ok { get; init; }
	public int Files { get; init; }
	public int Cached { get; init; }
	public int Failed { get; init; }
	public int TestsPassed { get; init; }
	public int TestsFailed { get; init; }
	public List<FileReport> FilesReports { get; init; } = [];
}

public static class ScrunchAnalyzer
{
	public static FileReport AnalyzeFile(Package root, string filePath, bool force = false)
	{
		filePath = Path.GetFullPath(filePath);
		if (!File.Exists(filePath))
			return new FileReport
			{
				Path = filePath, Ok = false,
				Problems = [new Problem { Line = 0, Kind = "error", Message = "File not found" }]
			};
		if (!force && StrictBinaryCache.IsFresh(filePath))
			return new FileReport { Path = filePath, Cached = true, Ok = true };
		var lines = TypeLines.FromFile(filePath);
		var problems = new List<Problem>();
		var tests = new List<TestNotificationMessage>();
		try
		{
			var package = PackageResolver.Resolve(root, filePath);
			var type = package.SynchronizeAndGetType(Path.GetFileNameWithoutExtension(filePath), lines);
			if (type is { IsTrait: false })
			{
				var methods = ParseMethods(type.Methods);
				new TestRunner(package, null, methods, null, tests).Run(new VirtualMachine(package));
			}
		}
		catch (Exception exception)
		{
			problems.Add(new Problem
			{
				Line = DiagnosticFormatter.GetLineNumber(exception, lines.Length) + 1,
				Kind = "error",
				Message = DiagnosticFormatter.FormatMessage(exception.GetType().Name,
					DiagnosticFormatter.BuildExceptionText(exception))
			});
		}
		foreach (var test in tests.Where(test => test.State == TestState.Red))
			problems.Add(new Problem
			{
				Line = test.LineNumber + 1, Kind = "test",
				Message = test.Message ?? test.Details ?? test.Expression ?? "test failed"
			});
		return new FileReport
		{
			Path = filePath, Ok = problems.Count == 0,
			TestsPassed = tests.Count(test => test.State == TestState.Green),
			TestsFailed = tests.Count(test => test.State == TestState.Red), Problems = problems
		};
	}

	public static FolderReport AnalyzePath(Package root, string path, bool force = false)
	{
		path = Path.GetFullPath(path);
		if (File.Exists(path))
		{
			var single = AnalyzeFile(root, path, force);
			return ToFolder(path, [single]);
		}
		if (!Directory.Exists(path))
			return new FolderReport
			{
				Path = path, Ok = false, Files = 0, Failed = 1,
				FilesReports =
				[
					new FileReport
					{
						Path = path, Ok = false,
						Problems = [new Problem { Line = 0, Kind = "error", Message = "Path not found" }]
					}
				]
			};
		var reports = new List<FileReport>();
		foreach (var file in Directory.GetFiles(path, "*" + Type.Extension, SearchOption.AllDirectories))
			if (!IsIgnored(file))
				reports.Add(AnalyzeFile(root, file, force));
		return ToFolder(path, reports);
	}

	public static FolderReport Status(string path)
	{
		path = Path.GetFullPath(path);
		var files = File.Exists(path)
			? [path]
			: Directory.Exists(path)
				? Directory.GetFiles(path, "*" + Type.Extension, SearchOption.AllDirectories).
					Where(file => !IsIgnored(file)).ToArray()
				: [];
		var reports = files.Select(file => new FileReport
		{
			Path = file, Cached = StrictBinaryCache.IsFresh(file),
			Ok = StrictBinaryCache.IsFresh(file)
		}).ToList();
		return ToFolder(path, reports);
	}

	private static FolderReport ToFolder(string path, List<FileReport> reports) =>
		new()
		{
			Path = path, Files = reports.Count, Cached = reports.Count(report => report.Cached),
			Failed = reports.Count(report => !report.Ok),
			TestsPassed = reports.Sum(report => report.TestsPassed),
			TestsFailed = reports.Sum(report => report.TestsFailed),
			Ok = reports.Count > 0 && reports.All(report => report.Ok), FilesReports = reports
		};

	private static bool IsIgnored(string file)
	{
		var normalized = file.Replace('\\', '/');
		return normalized.Contains("/bin/", StringComparison.OrdinalIgnoreCase) ||
			normalized.Contains("/obj/", StringComparison.OrdinalIgnoreCase) ||
			normalized.Contains("/node_modules/", StringComparison.OrdinalIgnoreCase) ||
			normalized.Contains("/.git/", StringComparison.OrdinalIgnoreCase);
	}

	private static IEnumerable<Method> ParseMethods(IEnumerable<Method> methods)
	{
		foreach (var method in methods.Where(method => !method.IsGeneric))
			if (method.GetBodyAndParseIfNeeded() is Body body)
				yield return body.Method;
	}
}
