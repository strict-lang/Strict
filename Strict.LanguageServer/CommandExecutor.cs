using System.Diagnostics;
using MediatR;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Client.Capabilities;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using OmniSharp.Extensions.LanguageServer.Protocol.Server;
using OmniSharp.Extensions.LanguageServer.Protocol.Window;
using OmniSharp.Extensions.LanguageServer.Protocol.Workspace;
using Strict.HighLevelRuntime;
using Strict.Language;

namespace Strict.LanguageServer;

//ncrunch: no coverage start
public class CommandExecutor(ILanguageServerFacade languageServer,
	StrictDocument document, Package package) : IExecuteCommandHandler
{
	private const string CommandName = "strict-vscode-client.run";
	private const string NotificationName = "testRunnerNotification";

	Task<Unit> IRequestHandler<ExecuteCommandParams, Unit>.Handle(
		ExecuteCommandParams request, CancellationToken cancellationToken)
	{
		var methodCall = request.Arguments?[0]?["label"]?.ToString() ??
			request.Arguments?[0]?.ToString();
		var uriText = request.Arguments?[1]?.ToString();
		try
		{
			var documentUri = DocumentUri.From(uriText ?? throw new PathCanNotBeEmpty());
			var localPath = documentUri.ToLocalFile();
			var folderName = localPath.GetFolderNameFromFile();
			var subPackage = package.Find(folderName) ?? new Package(package, folderName);
			RunAndNotify(documentUri, localPath, methodCall, subPackage);
		}
		catch (Exception exception)
		{
			languageServer.Window.LogError(exception.Message);
			languageServer.SendNotification(NotificationName, new TestNotificationMessage
			{
				LineNumber = 0,
				State = TestState.Red,
				Uri = uriText,
				MethodName = string.IsNullOrWhiteSpace(methodCall)
					? Method.Run
					: methodCall,
				Message = DiagnosticFormatter.FormatMessage(exception.GetType().Name,
					exception.Message),
				StackTrace = exception.ToString()
			});
		}
		return Unit.Task;
	}

	private void RunAndNotify(DocumentUri documentUri, string localPath, string? methodCall,
		Package subPackage)
	{
		var code = LinesFor(documentUri, localPath);
		var typeName = documentUri.Path.GetFileName();
		subPackage.LoadSiblingTypes(localPath, typeName);
		var type = subPackage.SynchronizeAndGetType(typeName, code);
		var methodName = string.IsNullOrWhiteSpace(methodCall)
			? Method.Run
			: methodCall.Trim();
		var method = type.Methods.FirstOrDefault(item => item.Name == methodName) ??
			type.Methods.FirstOrDefault(item => item.Name == Method.Run);
		languageServer.Window.LogInfo("SCrunch running " + type.Name + "." +
			(method?.Name ?? methodName));
		var started = Stopwatch.GetTimestamp();
		Exception? error = null;
		var consoleOutput = ConsoleCapture.Run(() =>
		{
			try
			{
				var interpreter = new Interpreter(package, TestBehavior.Disabled);
				if (methodName == Method.Run || method?.Name == Method.Run)
					interpreter.ExecuteRunMethod(type);
				else if (method != null)
					interpreter.Execute(method);
				else
					error = new InvalidOperationException("No runnable method named " + methodName);
			}
			catch (Exception exception)
			{
				error = exception;
			}
		});
		var durationMs = Stopwatch.GetElapsedTime(started).TotalMilliseconds;
		if (!string.IsNullOrWhiteSpace(consoleOutput))
			languageServer.Window.LogInfo(consoleOutput);
		if (error != null)
			languageServer.Window.LogError(error.Message);
		languageServer.SendNotification(NotificationName, new TestNotificationMessage
		{
			LineNumber = error is ParsingFailed parsing
				? parsing.FileLineNumber
				: method?.TypeLineNumber ?? 0,
			State = error == null
				? TestState.Green
				: TestState.Red,
			Uri = documentUri.ToString(),
			MethodName = method?.Name ?? methodName,
			TypeName = type.Name,
			DurationMs = durationMs,
			ConsoleOutput = string.IsNullOrWhiteSpace(consoleOutput)
				? null
				: consoleOutput,
			Message = error == null
				? null
				: DiagnosticFormatter.FormatMessage(error.GetType().Name, error.Message),
			StackTrace = error?.ToString(),
			MethodsCalled = 1
		});
	}

	private string[] LinesFor(DocumentUri documentUri, string localPath)
	{
		var code = document.Get(documentUri);
		if ((code.Length > 1 || code[0].Length > 0) || !File.Exists(localPath))
			return code;
		return File.ReadAllLines(localPath);
	}

	public ExecuteCommandRegistrationOptions GetRegistrationOptions(
		ExecuteCommandCapability capability, ClientCapabilities clientCapabilities) =>
		new() { Commands = new Container<string>(CommandName) };
}