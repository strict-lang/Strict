using MediatR;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Client.Capabilities;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using OmniSharp.Extensions.LanguageServer.Protocol.Server;
using OmniSharp.Extensions.LanguageServer.Protocol.Window;
using OmniSharp.Extensions.LanguageServer.Protocol.Workspace;
using Strict.Language;
using Strict.Expressions;
using Strict.Bytecode;

namespace Strict.LanguageServer;

//ncrunch: no coverage start
public class CommandExecutor(ILanguageServerFacade languageServer,
	StrictDocument document, Package package) : IExecuteCommandHandler
{
	private const string CommandName = "strict-vscode-client.run";

	Task<Unit> IRequestHandler<ExecuteCommandParams, Unit>.Handle(
		ExecuteCommandParams request, CancellationToken cancellationToken)
	{
		var methodCall = request.Arguments?[0]["label"]?.ToString() ?? null;
		var documentUri =
			DocumentUri.From(request.Arguments?[1].ToString() ?? throw new PathCanNotBeEmpty());
		var folderName = documentUri.Path.GetFolderName();
		var subPackage = package.Find(folderName) ?? new Package(package, folderName);
		var returns = AddAndExecute(documentUri, methodCall, subPackage);
		if (returns != null)
			languageServer.Window.LogInfo($"Output: {returns.Value}");
		return Unit.Task;
	}

	private ValueInstance? AddAndExecute(DocumentUri documentUri, string? methodCall, Package subPackage)
	{
		var code = document.Get(documentUri);
		var typeName = documentUri.Path.GetFileName();
		var type = subPackage.SynchronizeAndGetType(typeName, code);
		var call = (MethodCall)type.ParseExpression(methodCall);
		var binary = new BinaryGenerator(call).Generate();
		languageServer.Window.LogInfo($"Compiling: {
			Environment.NewLine + string.Join(",",
				binary.EntryPoint.instructions.Select(instruction => instruction + Environment.NewLine))
		}");
		return new VirtualMachine(binary).Execute().Returns;
	}

	public ExecuteCommandRegistrationOptions GetRegistrationOptions(
		ExecuteCommandCapability capability, ClientCapabilities clientCapabilities) =>
		new() { Commands = new Container<string>(CommandName) };
}