using MediatR;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Client.Capabilities;
using OmniSharp.Extensions.LanguageServer.Protocol.Document;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using OmniSharp.Extensions.LanguageServer.Protocol.Server;
using OmniSharp.Extensions.LanguageServer.Protocol.Server.Capabilities;
using OmniSharp.Extensions.LanguageServer.Protocol.Window;
using Strict.Language;

namespace Strict.LanguageServer;

public sealed class TextDocumentSynchronizer : ITextDocumentSyncHandler
{
	public TextDocumentSynchronizer(ILanguageServerFacade languageServer, StrictDocument document, Package package)
	{
		this.languageServer = languageServer;
		Document = document;
		this.package = package;
	}

	public StrictDocument Document { get; }
	private readonly ILanguageServerFacade languageServer;
	private readonly Package package;
	public TextDocumentAttributes GetTextDocumentAttributes(DocumentUri uri) => new(uri, "strict");

	public Task<Unit> Handle(DidChangeTextDocumentParams request,
		CancellationToken cancellationToken)
	{
		var uri = request.TextDocument.Uri.ToString();
		if (!Document.Contains(uri))
			Document.AddOrUpdate(uri, request.ContentChanges.ToArray().Select(x => x.Text).ToArray());
		Document.Update(uri, request.ContentChanges.ToArray());
		languageServer.Window.LogInfo($"Updated document: {uri}\n{Document.Get(uri)[^1]}");
		ParseUpdatedCodeAndPublishDiagnostics(request, uri);
		return Unit.Task;
	}

	private void
		ParseUpdatedCodeAndPublishDiagnostics(DidChangeTextDocumentParams request, string uri) =>
		languageServer.TextDocument.PublishDiagnostics(new PublishDiagnosticsParams
		{
			Diagnostics =
				new Container<Diagnostic>(Document.GetDiagnostics(package, request.TextDocument.Uri,
					languageServer)),
			Uri = uri,
			Version = 1
		});

	public Task<Unit> Handle(DidOpenTextDocumentParams request, CancellationToken cancellationToken)
	{
		Document.AddOrUpdate(request.TextDocument.Uri, request.TextDocument.Text.Split("\r\n"));
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
		new()
		{
			DocumentSelector = BaseSelectors.StrictDocumentSelector,
			SyncKind = TextDocumentSyncKind.Incremental
		};

	TextDocumentSaveRegistrationOptions
		IRegistration<TextDocumentSaveRegistrationOptions, SynchronizationCapability>.
		GetRegistrationOptions(SynchronizationCapability capability,
			ClientCapabilities clientCapabilities) =>
		new() { DocumentSelector = BaseSelectors.StrictDocumentSelector, IncludeText = true };

	TextDocumentOpenRegistrationOptions
		IRegistration<TextDocumentOpenRegistrationOptions, SynchronizationCapability>.
		GetRegistrationOptions(SynchronizationCapability capability,
			ClientCapabilities clientCapabilities) =>
		new() { DocumentSelector = BaseSelectors.StrictDocumentSelector };

	TextDocumentCloseRegistrationOptions
		IRegistration<TextDocumentCloseRegistrationOptions, SynchronizationCapability>.
		GetRegistrationOptions(SynchronizationCapability capability,
			ClientCapabilities clientCapabilities) =>
		new() { DocumentSelector = BaseSelectors.StrictDocumentSelector };
}