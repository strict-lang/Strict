using MediatR;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Client.Capabilities;
using OmniSharp.Extensions.LanguageServer.Protocol.Document;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using OmniSharp.Extensions.LanguageServer.Protocol.Server;
using OmniSharp.Extensions.LanguageServer.Protocol.Server.Capabilities;
using OmniSharp.Extensions.LanguageServer.Protocol.Window;

namespace Strict.LanguageServer;

// ReSharper disable once HollowTypeName
public sealed class TextDocumentSyncHandler : ITextDocumentSyncHandler
{
	public TextDocumentSyncHandler(ILanguageServerFacade languageServer) => this.languageServer = languageServer;
	private readonly StrictDocumentManager documentManager = new();
	private readonly ILanguageServerFacade languageServer;
	private readonly DocumentSelector documentSelector = new(
		new DocumentFilter
		{
			Pattern = "**/*.strict"
		}
	);
	public TextDocumentAttributes GetTextDocumentAttributes(DocumentUri uri) => new(uri, "strict");

	public Task<Unit> Handle(DidChangeTextDocumentParams request, CancellationToken cancellationToken)
	{
		var documentPath = request.TextDocument.Uri.ToString();
		var text = request.ContentChanges.ToArray()[0].Text;
		documentManager.Update(documentPath, text);
		languageServer.Window.LogInfo($"Updated document: {documentPath}\n{text}");
		return Unit.Task;
	}

	public Task<Unit> Handle(DidOpenTextDocumentParams request, CancellationToken cancellationToken)
	{
		documentManager.Update(request.TextDocument.Uri, request.TextDocument.Text);
		return Unit.Task;
	}

	public Task<Unit> Handle(DidCloseTextDocumentParams request,
		CancellationToken cancellationToken) =>
		throw new NotImplementedException();

	public Task<Unit>
		Handle(DidSaveTextDocumentParams request, CancellationToken cancellationToken) =>
		throw new NotImplementedException();

	public TextDocumentChangeRegistrationOptions GetRegistrationOptions(
		SynchronizationCapability capability, ClientCapabilities clientCapabilities) =>
		new() { DocumentSelector = documentSelector, SyncKind = TextDocumentSyncKind.Incremental };

	TextDocumentSaveRegistrationOptions
		IRegistration<TextDocumentSaveRegistrationOptions, SynchronizationCapability>.
		GetRegistrationOptions(SynchronizationCapability capability,
			ClientCapabilities clientCapabilities) =>
		new()
		{
			DocumentSelector = documentSelector, IncludeText = true
		};

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