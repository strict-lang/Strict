using System.Collections.Concurrent;
using System.Text;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

/// <summary>
/// Strict File handles only remember their path, each operation opens the file just for itself, so
/// other processes (parallel tests, editors) can always read the same files at the same time.
/// </summary>
public static class NativeFileRegistry
{
	private static readonly UTF8Encoding Utf8WithoutBom = new(false);
	private static long nextHandle = 1;
	private static readonly ConcurrentDictionary<long, string> OpenFiles = new();

	public static ValueInstance Open(Type fileType, string path)
	{
		var handle = Interlocked.Increment(ref nextHandle);
		OpenFiles[handle] = path;
		return new ValueInstance(fileType, handle);
	}

	/// <summary>
	/// Reading never creates a file, a missing one throws FileNotFoundException.
	/// </summary>
	private static FileStream OpenForReading(long handle) =>
		new(GetPath(handle), FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);

	public static string ReadText(long handle)
	{
		using var reader = new StreamReader(OpenForReading(handle), Utf8WithoutBom, true);
		return reader.ReadToEnd();
	}

	public static string[] ReadLines(long handle) =>
		ReadText(handle).Replace("\r", string.Empty, StringComparison.Ordinal).Split('\n');

	public static byte[] ReadBytes(long handle)
	{
		using var stream = OpenForReading(handle);
		var bytes = new byte[stream.Length];
		stream.ReadExactly(bytes);
		return bytes;
	}

	public static void WriteText(long handle, string text) =>
		File.WriteAllText(GetPath(handle), text, Utf8WithoutBom);

	public static void WriteLines(long handle, IEnumerable<string> lines) =>
		WriteText(handle, string.Join('\n', lines));

	public static void WriteBytes(long handle, byte[] bytes) =>
		File.WriteAllBytes(GetPath(handle), bytes);

	public static void Delete(long handle)
	{
		var path = GetPath(handle);
		Close(handle);
		if (File.Exists(path))
			File.Delete(path);
	}

	public static void Close(long handle) => OpenFiles.TryRemove(handle, out _);

	public static bool Exists(long handle) =>
		OpenFiles.TryGetValue(handle, out var path) && File.Exists(path);

	public static long Length(long handle) => new FileInfo(GetPath(handle)).Length;

	private static string GetPath(long handle) =>
		OpenFiles.TryGetValue(handle, out var path)
			? path
			: throw new FileHandleNotOpen(handle);

	public sealed class FileHandleNotOpen(long handle) : Exception("File handle not open: " + handle);
}
