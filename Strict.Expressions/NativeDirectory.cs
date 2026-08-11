namespace Strict.Expressions;

/// <summary>
/// Directory filesystem helpers used by the Strict VM for Directory.strict.
/// </summary>
public static class NativeDirectory
{
	public static bool Exists(string path) =>
		!string.IsNullOrWhiteSpace(path) && Directory.Exists(path);

	public static void Create(string path)
	{
		if (!string.IsNullOrWhiteSpace(path))
			Directory.CreateDirectory(path);
	}

	public static string[] GetFiles(string path, string pattern)
	{
		if (!Directory.Exists(path))
			return [];
		return string.IsNullOrEmpty(pattern)
			? Directory.GetFiles(path)
			: Directory.GetFiles(path, pattern);
	}
}
