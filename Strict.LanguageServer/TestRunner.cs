using System.Diagnostics;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Server;
using Strict.HighLevelRuntime;
using Strict.Language;

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
}