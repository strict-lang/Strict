using OmniSharp.Extensions.LanguageServer.Protocol.Server;
using Strict.Language;
using Strict.Expressions;
using Strict.Bytecode;

namespace Strict.LanguageServer;

//ncrunch: no coverage start
public sealed class TestRunner(Package package, ILanguageServerFacade languageServer,
	IEnumerable<Method> methods) : RunnerService(package), RunnableService
{
	private IEnumerable<Method> Methods { get; } = methods;
	private const string NotificationName = "testRunnerNotification";

	public void Run(VirtualMachine vm)
	{
		foreach (var test in Methods.SelectMany(method => method.Tests))
			if (test is MethodCall { Instance: { } } methodCall)
			{
				var binary = new BinaryGenerator((MethodCall)methodCall.Instance).Generate();
				var output = new VirtualMachine(binary).ExecuteRun().Returns;
				languageServer?.SendNotification(NotificationName, new TestNotificationMessage(
					GetLineNumber(test), Equals(output, ((Value)methodCall.Arguments[0]).Data)
						? TestState.Green
						: TestState.Red));
			}
	}

	private int GetLineNumber(Expression test)
	{
		foreach (var method in Methods)
		{
			var line = method.Tests.FindIndex(testToFind => testToFind.ToString() == test.ToString());
			if (line != -1)
				return method.TypeLineNumber + line + 1;
		}
		throw new KeyNotFoundException();
	}
}