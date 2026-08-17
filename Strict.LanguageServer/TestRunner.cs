using System.Diagnostics;
using OmniSharp.Extensions.LanguageServer.Protocol.Server;
using Strict.HighLevelRuntime;
using Strict.Language;
using Strict.TestRunner;
using VirtualMachine = Strict.VirtualMachine;

namespace Strict.LanguageServer;

//ncrunch: no coverage start
public sealed class TestRunner(Package package, ILanguageServerFacade languageServer,
	IEnumerable<Method> methods, string? uri = null) : RunnerService(package), RunnableService
{
	private IEnumerable<Method> Methods { get; } = methods;
	private const string NotificationName = "testRunnerNotification";

	public void Run(VirtualMachine vm)
	{
		var first = Methods.FirstOrDefault(item => item.Tests.Count > 0);
		if (first == null)
			return;
		var interpreter = new TestInterpreter(first.Type.Package);
		foreach (var method in Methods.Where(item => item.Tests.Count > 0))
			RunMethod(interpreter, method);
	}

	private void RunMethod(TestInterpreter interpreter, Method method)
	{
		interpreter.Statistics.Reset();
		var started = Stopwatch.GetTimestamp();
		Interpreter.TestFailed? failed = null;
		Exception? executionError = null;
		var consoleOutput = ConsoleCapture.Run(() =>
		{
			try
			{
				interpreter.Execute(method);
			}
			catch (Interpreter.TestFailed exception)
			{
				failed = exception;
			}
			catch (Exception exception)
			{
				executionError = exception;
			}
		});
		var durationMs = Stopwatch.GetElapsedTime(started).TotalMilliseconds;
		if (executionError != null)
		{
			NotifyExecutionError(method, durationMs, consoleOutput, executionError);
			return;
		}
		NotifyMethod(method, interpreter, durationMs, consoleOutput, failed);
	}

	private void NotifyExecutionError(Method method, double durationMs, string consoleOutput,
		Exception error)
	{
		var line = method.Tests.Count > 0
			? method.Tests[0].LineNumber
			: method.TypeLineNumber;
		languageServer.SendNotification(NotificationName, new TestNotificationMessage
		{
			LineNumber = line,
			State = TestState.Red,
			Uri = uri,
			Expression = method.Tests.Count > 0
				? method.Tests[0].ToString()
				: method.Name,
			MethodName = method.Name,
			TypeName = method.Type.Name,
			DurationMs = durationMs,
			ConsoleOutput = string.IsNullOrWhiteSpace(consoleOutput)
				? null
				: consoleOutput,
			Message = error.Message,
			StackTrace = error.ToString()
		});
	}

	private void NotifyMethod(Method method, TestInterpreter interpreter, double durationMs,
		string consoleOutput, Interpreter.TestFailed? failed)
	{
		var tests = method.Tests;
		var failedIndex = failed == null
			? -1
			: tests.FindIndex(test => test.LineNumber == failed.FailedExpression.LineNumber ||
				test.ToString() == failed.FailedExpression.ToString());
		if (failed != null && failedIndex < 0)
			failedIndex = tests.Count - 1;
		var perTest = tests.Count > 0
			? durationMs / tests.Count
			: durationMs;
		for (var index = 0; index < tests.Count; index++)
		{
			if (failedIndex >= 0 && index > failedIndex)
				break;
			var test = tests[index];
			var isFailed = index == failedIndex;
			languageServer.SendNotification(NotificationName, new TestNotificationMessage
			{
				LineNumber = test.LineNumber,
				State = isFailed
					? TestState.Red
					: TestState.Green,
				Uri = uri,
				Expression = test.ToString(),
				MethodName = method.Name,
				TypeName = method.Type.Name,
				DurationMs = perTest,
				ConsoleOutput = string.IsNullOrWhiteSpace(consoleOutput)
					? null
					: consoleOutput,
				MethodsCalled = interpreter.Statistics.MethodCallCount,
				LinesCalled = interpreter.Statistics.ExpressionCount,
				CallCount = tests.Count,
				Details = isFailed
					? failed?.Details
					: null,
				Expected = isFailed
					? SplitIs(failed?.Details).expected
					: null,
				Actual = isFailed
					? SplitIs(failed?.Details).actual
					: null,
				Message = isFailed
					? failed?.Message
					: null,
				StackTrace = isFailed && failed != null
					? StackFrom(method, failed)
					: null
			});
		}
	}

	private static (string? expected, string? actual) SplitIs(string? details)
	{
		if (string.IsNullOrEmpty(details))
			return (null, null);
		var separator = details.LastIndexOf(" is ", StringComparison.Ordinal);
		return separator < 0
			? (null, null)
			: (details[(separator + 4)..].Trim(), details[..separator].Trim());
	}

	private static string StackFrom(Method method, Interpreter.TestFailed failed) =>
		"at " + method.Type.FullName + "." + method.Name + " in " + method.Type.FilePath +
		":line " + (failed.FailedExpression.LineNumber + 1);
}