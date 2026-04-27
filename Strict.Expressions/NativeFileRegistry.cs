using System.Collections.Concurrent;
using System.Text;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public static class NativeFileRegistry
{
	private static readonly UTF8Encoding Utf8WithoutBom = new(false);

	private sealed class FileState(string path, FileStream stream)
	{
		public readonly string Path = path;
		public readonly FileStream Stream = stream;
	}

	private static long nextHandle = 1;
	private static readonly ConcurrentDictionary<long, FileState> OpenFiles = new();

	public static ValueInstance Open(Type fileType, string path)
	{
		var handle = Interlocked.Increment(ref nextHandle);
		var stream = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite,
			FileShare.ReadWrite);
		OpenFiles[handle] = new FileState(path, stream);
		return new ValueInstance(fileType, handle);
	}

	public static string ReadText(long handle)
	{
		var state = Get(handle);
		state.Stream.Position = 0;
		using var reader = new StreamReader(state.Stream, Utf8WithoutBom, true, 1024, true);
		var text = reader.ReadToEnd();
		state.Stream.Position = 0;
		return text;
	}

	public static byte[] ReadBytes(long handle)
	{
		var state = Get(handle);
		state.Stream.Position = 0;
		var bytes = new byte[(int)state.Stream.Length];
		state.Stream.ReadExactly(bytes);
		state.Stream.Position = 0;
		return bytes;
	}

	public static void WriteText(long handle, string text)
	{
		var state = Get(handle);
		state.Stream.SetLength(0);
		state.Stream.Position = 0;
		using var writer = new StreamWriter(state.Stream, Utf8WithoutBom, 1024, true);
		writer.Write(text);
		writer.Flush();
		state.Stream.Position = 0;
	}

	public static void WriteBytes(long handle, byte[] bytes)
	{
		var state = Get(handle);
		state.Stream.SetLength(0);
		state.Stream.Position = 0;
		state.Stream.Write(bytes);
		state.Stream.Flush();
		state.Stream.Position = 0;
	}

	public static void Delete(long handle)
	{
		var path = Get(handle).Path;
		Close(handle);
		if (File.Exists(path))
			File.Delete(path);
	}

	public static void Close(long handle)
	{
		if (!OpenFiles.TryRemove(handle, out var state))
			return;
		state.Stream.Dispose();
	}

	public static bool Exists(long handle)
	{
		if (!OpenFiles.TryGetValue(handle, out var state))
			return false;
		return File.Exists(state.Path);
	}

	public static long Length(long handle) => Get(handle).Stream.Length;

	private static FileState Get(long handle) =>
		OpenFiles.TryGetValue(handle, out var state)
			? state
			: throw new InvalidOperationException("File handle not open: " + handle);
}
