// When things are in flux, force generating a new .strictbinary every time by disabling the cache
//#define DISABLE_BINARY_CACHE
using Strict.Bytecode;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict;

public sealed partial class Runner
{
	private static bool UsedPackageHasNewerStrictFile(BinaryExecutable binary, DateTime binaryTime)
	{
		var strictRoot = Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict));
		return binary.MethodsPerType.Keys.Select(GetPackageName).Where(name => name.Length > 0).
			Distinct().Any(packageName => DirectoryHasNewerFile(Path.Combine(strictRoot,
				Path.GetRelativePath(nameof(Strict), packageName)), "*" + Type.Extension, binaryTime));
	}

	private static string GetPackageName(string typeFullName)
	{
		var nonGenericName = typeFullName.Split('(')[0];
		var separatorIndex = nonGenericName.LastIndexOf(Context.ParentSeparator);
		return separatorIndex > 0 && nonGenericName.StartsWith(nameof(Strict), StringComparison.Ordinal)
			? nonGenericName[..separatorIndex]
			: "";
	}

	/// <summary>
	/// A newer .strict file or Strict*.dll outdates the cache. Enumerated files already carry their
	/// write time, no extra file system call per file is needed.
	/// </summary>
	private static bool DirectoryHasNewerFile(string? directory, string searchPattern,
		DateTime binaryTime) =>
		!string.IsNullOrEmpty(directory) && Directory.Exists(directory) && new DirectoryInfo(directory).
			EnumerateFiles(searchPattern).Any(file => file.LastWriteTimeUtc > binaryTime);

	private BinaryExecutable CacheStrictExecutable(BinaryExecutable binary)
	{
		var outputFilePath = Path.ChangeExtension(strictFilePath, BinaryExecutable.Extension);
		try
		{
			binary.Serialize(outputFilePath);
			Log("Saving " + new FileInfo(outputFilePath).Length + " bytes of bytecode to: " +
				outputFilePath);
		}
		catch (BinaryExecutable.ValueInstanceNotSupported ex)
		{
			Log("Bytecode serialization not yet supported for this program: " + ex.Message);
		}
		catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
		{
			Log("Cached binary not saved, it is in use: " + ex.Message);
		}
		return binary;
	}
}
