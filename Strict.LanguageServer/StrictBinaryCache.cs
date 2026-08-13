using Strict.Bytecode;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.LanguageServer;

public static class StrictBinaryCache
{
	public static bool IsFresh(long sourceTimeTicks, long? binaryTimeTicks) =>
		binaryTimeTicks != null && binaryTimeTicks.Value >= sourceTimeTicks;

	public static bool IsFresh(string sourcePath)
	{
		var binaryPath = Path.ChangeExtension(sourcePath, BinaryExecutable.Extension);
		if (!File.Exists(sourcePath) || !File.Exists(binaryPath))
			return false;
		return IsFresh(File.GetLastWriteTimeUtc(sourcePath).Ticks,
			File.GetLastWriteTimeUtc(binaryPath).Ticks);
	}

	public static void TrySaveAfterPassingTests(Type type)
	{
		var sourcePath = type.FilePath;
		if (string.IsNullOrEmpty(sourcePath) || !File.Exists(sourcePath))
			return;
		var runMethods = type.Methods.Where(method => method.Name == Method.Run).ToArray();
		if (runMethods.Length == 0)
			return;
		try
		{
			var preferred = runMethods.FirstOrDefault(method => method.Parameters.Count == 0) ??
				runMethods[0];
			var binary = BinaryGenerator.GenerateFromRunMethods(preferred, runMethods);
			binary.Serialize(Path.ChangeExtension(sourcePath, BinaryExecutable.Extension));
		}
		catch (Exception)
		{
			// Tests already passed; not every type can emit a .strictbinary yet.
		}
	}
}
