using MediatR;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Client.Capabilities;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using OmniSharp.Extensions.LanguageServer.Protocol.Server;
using OmniSharp.Extensions.LanguageServer.Protocol.Window;
using OmniSharp.Extensions.LanguageServer.Protocol.Workspace;
using Strict.Language;
using Strict.Language.Expressions;
using Strict.VirtualMachine;

namespace Strict.LanguageServer;
//ncrunch: no coverage start

public class CommandExecutor : IExecuteCommandHandler
{
	private readonly VirtualMachine.VirtualMachine vm = new();
	private const string CommandName = "strict-vscode-client.run";
	private readonly StrictDocument document;
	private readonly ILanguageServerFacade languageServer;
	private readonly Package package;

	public CommandExecutor(ILanguageServerFacade languageServer, StrictDocument document,
		Package package)
	{
		this.document = document;
		this.languageServer = languageServer;
		this.package = package;
	}

	Task<Unit> IRequestHandler<ExecuteCommandParams, Unit>.Handle(
		ExecuteCommandParams request, CancellationToken cancellationToken)
	{
		var methodCall = request.Arguments?[0]["label"].ToString();
		var documentUri =
			DocumentUri.From(request.Arguments?[1].ToString() ?? throw new PathCanNotBeEmpty());
		var folderName = documentUri.Path.GetFolderName();
		var subPackage = package.Find(folderName) ?? new Package(package, folderName);
		AddAndExecute(documentUri, methodCall, subPackage);
		if (vm.Returns != null)
			languageServer.Window.LogInfo($"Output: {vm.Returns.Value}");
		return Unit.Task;
	}

	private void AddAndExecute(DocumentUri documentUri, string? methodCall, Package subPackage)
	{
		var code = document.Get(documentUri);
		var typeName = documentUri.Path.GetFileName();
		var type = subPackage.SynchronizeAndGetType(typeName, code);
		var call = (MethodCall)type.ParseExpression(methodCall);
		var statements = new ByteCodeGenerator(call).Generate();
		languageServer.Window.LogInfo(
			$"Compiling : \n{string.Join(",", statements.ConvertAll(statement => statement + Environment.NewLine))}");
		vm.Execute(statements);
	}

	public ExecuteCommandRegistrationOptions GetRegistrationOptions(
		ExecuteCommandCapability capability, ClientCapabilities clientCapabilities) =>
		new() { Commands = new Container<string>(CommandName) };
}

public class PathCanNotBeEmpty : Exception { }