#if DEBUG
namespace Strict.Language;

public static class PerformanceLog
{
	public const string EnvironmentVariableName = "STRICT_PERFORMANCE_LOGGING";
	public static bool IsEnabled { get; set; } =
		Environment.GetEnvironmentVariable(EnvironmentVariableName) is "1" or "true" or "TRUE";
	public static bool IsTemporarilyDisabled => suppressedDepth > 0;

	public static void Write(string source, string message)
	{
		if (!IsEnabled || suppressedDepth > 0)
			return;
		LogWriter.WriteLine(source + " " + message);
		LogWriter.Flush();
	}

	public static IDisposable Suppress() => new SuppressPerformanceLogging();

	public static string GetCallers(int skipFrames = 0, int methodCount = 8)
	{
		var stackTrace = new System.Diagnostics.StackTrace(skipFrames + 1, false);
		var frames = stackTrace.GetFrames();
		if (frames.Length == 0)
			return "unknown";
		var callers = new List<string>(methodCount);
		for (var frameIndex = 0; frameIndex < frames.Length && callers.Count < methodCount;
			frameIndex++)
		{
			var method = frames[frameIndex].GetMethod();
			var declaringType = method?.DeclaringType;
			if (method == null || declaringType == typeof(PerformanceLog))
				continue;
			callers.Add((declaringType?.Name ?? "Unknown") + "." + method.Name);
		}
		return callers.Count == 0
			? "unknown"
			: string.Join(" <- ", callers);
	}

	public static TextWriter LogWriter
	{
		get	
		{
			if (field != null)
				return field;
			var logFile = new FileStream("Strict.txt", FileMode.Create, FileAccess.Write, FileShare.ReadWrite);
			field = new StreamWriter(logFile);
			return field;
		}
	}
	[ThreadStatic]
	private static int suppressedDepth;

	private sealed class SuppressPerformanceLogging : IDisposable
	{
		public SuppressPerformanceLogging() => suppressedDepth++;
		public void Dispose() => suppressedDepth--;
	}
}
#endif