using System.Diagnostics;
using System.Diagnostics;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Server;
using Strict.HighLevelRuntime;
using Strict.Language;
using Strict.TestRunner;
using VirtualMachine = Strict.VirtualMachine;

namespace Strict.LanguageServer;

//ncrunch: no coverage start
public sealed class TestRunner(Package package, ILanguageServerFacade? languageServer,
	IEnumerable<Method> methods, DocumentUri? documentUri = null,
	ICollection<TestNotificationMessage>? sink = null) : RunnerService(package),
	RunnableService
{
	private IEnumerable<Method> Methods { get; } = methods;
	private readonly string? uri = documentUri?.ToString();
	private const string NotificationName = "testRunnerNotification";

	public void Run(VirtualMachine vm)
	{
var methodList = Methods.ToList();
		if (methodList.Count == 0)
			return;
		var interpreter = new Interpreter(methodList[0].Type.Package, TestBehavior.TestRunner);
		var allPassed = true;
		foreach (var method in methodList)
			if (!RunMethodTests(interpreter, method))
				allPassed = false;
		if (allPassed)
			StrictBinaryCache.TrySaveAfterPassingTests(methodList[0].Type);
	}

	private bool RunMethodTests(Interpreter interpreter, Method method)
	{
		var tests = method.Tests;
		if (tests.Count == 0)
			return true;
		var watch = Stopwatch.StartNew();
		try
		{
			interpreter.Execute(method);
			watch.Stop();
			foreach (var test in tests)
				Notify(test, method, TestState.Green, null, null, watch.Elapsed.TotalMilliseconds);
			return true;
		}
		catch (Interpreter.TestFailed failed)
		{
			watch.Stop();
			var failedText = failed.FailedExpression.ToString();
			var seenFailed = false;
			var durationMs = watch.Elapsed.TotalMilliseconds;
			var stack = StackFrom(failed.Message);
			foreach (var test in tests)
			{
				if (!seenFailed && test.ToString() == failedText)
				{
					Notify(test, method, TestState.Red, failed.Message, failed.Details, durationMs,
						stack);
					seenFailed = true;
					continue;
				}
				Notify(test, method, seenFailed
					? TestState.Red
					: TestState.Green, seenFailed
					? failed.Message
					: null, null, durationMs, seenFailed
					? stack
					: null);
			}
			return false;
		}
		catch (Exception exception)
		{
			watch.Stop();
			var text = DiagnosticFormatter.BuildExceptionText(exception);
			foreach (var test in tests)
				Notify(test, method, TestState.Red, text, null, watch.Elapsed.TotalMilliseconds,
					StackFrom(text));
			return false;
		}
	}

	private void Notify(Expression test, Method method, TestState state, string? message,
		string? details, double durationMs, string? stackTrace = null)
	{
		var notification = new TestNotificationMessage(GetLineNumber(test, method), state, uri,
			test.ToString(), method.Name, message, details, durationMs, stackTrace, method.Type.Name);
		sink?.Add(notification);
		languageServer?.SendNotification(NotificationName, notification);
	}

	private static string? StackFrom(string? message)
	{
		if (string.IsNullOrEmpty(message))
			return null;
		var atIndex = message.IndexOf("\n   at ", StringComparison.Ordinal);
		return atIndex >= 0
			? message[atIndex..].Trim()
			: null;
	}

	private static int GetLineNumber(Expression test, Method method)
	{
		if (test.LineNumber > 0)
			return test.LineNumber;
		var index = method.Tests.FindIndex(candidate => candidate.ToString() == test.ToString());
		return index == -1
			? method.TypeLineNumber
			: method.TypeLineNumber + index + 1;
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