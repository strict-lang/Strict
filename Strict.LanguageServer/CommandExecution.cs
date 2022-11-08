using MediatR;
using OmniSharp.Extensions.LanguageServer.Protocol.Client.Capabilities;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using OmniSharp.Extensions.LanguageServer.Protocol.Workspace;

namespace Strict.LanguageServer
{
	public class CommandExecution : IExecuteCommandHandler
	{
		public Task<Unit> Handle(ExecuteCommandParams request, CancellationToken cancellationToken) => Unit.Task;

		public ExecuteCommandRegistrationOptions GetRegistrationOptions(
			ExecuteCommandCapability capability, ClientCapabilities clientCapabilities) =>
			new() { Commands = new Container<string>("strict-vscode-client.run") };
	}
}