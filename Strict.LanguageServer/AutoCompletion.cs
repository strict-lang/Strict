using OmniSharp.Extensions.LanguageServer.Protocol.Client.Capabilities;
using OmniSharp.Extensions.LanguageServer.Protocol.Document;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.LanguageServer;

public sealed class AutoCompletion : ICompletionHandler
{
	private readonly StrictDocument documentManager;

	public AutoCompletion(StrictDocument documentManager) =>
		this.documentManager = documentManager;

	public Task<CompletionList> Handle(CompletionParams request,
		CancellationToken cancellationToken)
	{
		if (request.Context?.TriggerCharacter == null)
			return Task.FromResult(new CompletionList());
		if (request.Context?.TriggerCharacter == ".")
			documentManager.Get(request.TextDocument.Uri);
		return Task.FromResult(new CompletionList());
	}

	public CompletionRegistrationOptions GetRegistrationOptions(CompletionCapability capability,
		ClientCapabilities clientCapabilities) =>
		new()
		{
			TriggerCharacters = new Container<string>("."),
			DocumentSelector = BaseSelectors.StrictDocumentSelector,
			ResolveProvider = true
		};
}