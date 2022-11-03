using MediatR;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Client.Capabilities;
using OmniSharp.Extensions.LanguageServer.Protocol.Document;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using OmniSharp.Extensions.LanguageServer.Protocol.Server;
using OmniSharp.Extensions.LanguageServer.Protocol.Server.Capabilities;
using OmniSharp.Extensions.LanguageServer.Protocol.Window;

namespace Strict.LanguageServer;

public sealed class TextDocumentSynchronizer : ITextDocumentSyncHandler
{
	public TextDocumentSynchronizer(ILanguageServerFacade languageServer) =>
		this.languageServer = languageServer;

	public readonly StrictDocumentManager DocumentManager = new();
	private readonly ILanguageServerFacade languageServer;
	private readonly DocumentSelector documentSelector = new(
		new DocumentFilter { Pattern = "**/*.strict" });
	public TextDocumentAttributes GetTextDocumentAttributes(DocumentUri uri) => new(uri, "strict");

	public Task<Unit> Handle(DidChangeTextDocumentParams request,
		CancellationToken cancellationToken)
	{
		var uri = request.TextDocument.Uri.ToString();
		DocumentManager.Update(uri, request.ContentChanges.ToArray());
		languageServer.Window.LogInfo($"Updated document: {uri}\n{DocumentManager.Get(uri)}");
		return Unit.Task;
	}

	public Task<Unit> Handle(DidOpenTextDocumentParams request, CancellationToken cancellationToken)
	{
		DocumentManager.AddOrUpdate(request.TextDocument.Uri, request.TextDocument.Text);
		return Unit.Task;
	}

	public Task<Unit> Handle(DidCloseTextDocumentParams request,
		CancellationToken cancellationToken) =>
		Unit.Task;

	public Task<Unit>
		Handle(DidSaveTextDocumentParams request, CancellationToken cancellationToken) =>
		Unit.Task;

	public TextDocumentChangeRegistrationOptions GetRegistrationOptions(
		SynchronizationCapability capability, ClientCapabilities clientCapabilities) =>
		new() { DocumentSelector = documentSelector, SyncKind = TextDocumentSyncKind.Incremental };

	TextDocumentSaveRegistrationOptions
		IRegistration<TextDocumentSaveRegistrationOptions, SynchronizationCapability>.
		GetRegistrationOptions(SynchronizationCapability capability,
			ClientCapabilities clientCapabilities) =>
		new() { DocumentSelector = documentSelector, IncludeText = true };

	TextDocumentOpenRegistrationOptions
		IRegistration<TextDocumentOpenRegistrationOptions, SynchronizationCapability>.
		GetRegistrationOptions(SynchronizationCapability capability,
			ClientCapabilities clientCapabilities) =>
		new() { DocumentSelector = documentSelector };

	TextDocumentCloseRegistrationOptions
		IRegistration<TextDocumentCloseRegistrationOptions, SynchronizationCapability>.
		GetRegistrationOptions(SynchronizationCapability capability,
			ClientCapabilities clientCapabilities) =>
		new() { DocumentSelector = documentSelector };
}