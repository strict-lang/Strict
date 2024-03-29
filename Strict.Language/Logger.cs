﻿#if LOG_DETAILS
namespace Strict.Language;

public sealed class Logger
{
	public static void Info(string message) =>
		Console.WriteLine($"{DateTime.UtcNow.ToShortTimeString()} {message}"); //ncrunch: no coverage
}
#endif