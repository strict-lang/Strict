// When things are in flux, force generating a new .strictbinary every time by disabling the cache
//#define DISABLE_BINARY_CACHE
using System.Globalization;
using Strict.Bytecode;
using Strict.Bytecode.Serialization;
using Strict.Compiler;
using Strict.Compiler.Assembly;
using Strict.Expressions;
using Strict.Language;
using Strict.Optimizers;
using Strict.TestRunner;
using Strict.Validators;
using Type = Strict.Language.Type;

namespace Strict;

public sealed partial class Runner
{
	private static bool UsedPackageHasNewerStrictFile(BinaryExecutable binary, DateTime binaryTime)
	{
		var strictRoot = Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict));
		return binary.MethodsPerType.Keys.Select(GetPackageName).Where(name => name.Length > 0).
			Distinct().Any(packageName => DirectoryHasNewerStrictFile(Path.Combine(strictRoot,
				Path.GetRelativePath(nameof(Strict), packageName)), binaryTime));
	}

	private static string GetPackageName(string typeFullName)
	{
		var nonGenericName = typeFullName.Split('(')[0];
		var separatorIndex = nonGenericName.LastIndexOf(Context.ParentSeparator);
		return separatorIndex > 0 && nonGenericName.StartsWith(nameof(Strict), StringComparison.Ordinal)
			? nonGenericName[..separatorIndex]
			: "";
	}

	private static bool DirectoryHasNewerStrictFile(string? directory, DateTime binaryTime)
	{
		if (string.IsNullOrEmpty(directory) || !Directory.Exists(directory))
			return false;
		foreach (var file in Directory.EnumerateFiles(directory, "*" + Type.Extension))
			if (File.GetLastWriteTimeUtc(file) > binaryTime)
				return true;
		return false;
	}

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
