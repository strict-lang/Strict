#if LOG_DETAILS
using System;

namespace Strict.Language.Expressions;

public sealed class Logger
{
	public static void Info(string message) =>
		Console.WriteLine($"{DateTime.UtcNow.ToShortTimeString()} {message}");
}
#endif