namespace Strict.LanguageServer;

internal static class ConsoleCapture
{
	public static string Run(Action action)
	{
		var writer = new StringWriter();
		var original = Console.Out;
		Console.SetOut(writer);
		try
		{
			action();
		}
		finally
		{
			Console.SetOut(original);
		}
		return writer.ToString();
	}
}
