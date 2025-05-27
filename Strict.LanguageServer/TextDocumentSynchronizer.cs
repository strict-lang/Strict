﻿using MediatR;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Client.Capabilities;
using OmniSharp.Extensions.LanguageServer.Protocol.Document;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using OmniSharp.Extensions.LanguageServer.Protocol.Server;
using OmniSharp.Extensions.LanguageServer.Protocol.Server.Capabilities;
using OmniSharp.Extensions.LanguageServer.Protocol.Window;
using Strict.Language;

namespace Strict.LanguageServer;

public sealed class TextDocumentSynchronizer(
	ILanguageServerFacade languageServer,
	StrictDocument document,
	Package strictBase) : ITextDocumentSyncHandler
{
	public StrictDocument Document { get; } = document;
	public TextDocumentAttributes GetTextDocumentAttributes(DocumentUri uri) => new(uri, "strict"); //ncrunch: no coverage

	public Task<Unit> Handle(DidChangeTextDocumentParams request,
		CancellationToken cancellationToken)
	{
		var uri = request.TextDocument.Uri.ToString();
		if (!Document.Contains(uri))
			Document.AddOrUpdate(uri, request.ContentChanges.ToArray().Select(x => x.Text).ToArray()); //ncrunch: no coverage
		Document.Update(uri, request.ContentChanges.ToArray());
		languageServer.Window.LogInfo($"Updated document: {uri}\n{Document.Get(uri)[^1]}");
		ParseUpdatedCodeAndPublishDiagnostics(request.TextDocument.Uri);
		return Unit.Task;
	}

	private void
		ParseUpdatedCodeAndPublishDiagnostics(DocumentUri uri) =>
		languageServer.TextDocument.PublishDiagnostics(new PublishDiagnosticsParams
		{
			Diagnostics =
				new Container<Diagnostic>(Document.GetDiagnostics(strictBase, uri,
					languageServer)),
			Uri = uri,
			Version = 1
		});

	//private Package? currentPackage;
	public Task<Unit> Handle(DidOpenTextDocumentParams request, CancellationToken cancellationToken)
	{
		//currentPackage ??=
		//	await new Repositories(new MethodExpressionParser()).LoadFromPath(string.Join("/",
		//		request.TextDocument.Uri.Path.Split('/')[..^1]));
		Document.AddOrUpdate(request.TextDocument.Uri, request.TextDocument.Text.Split("\r\n"));
		Document.InitializeContent(request.TextDocument.Uri);
		ParseUpdatedCodeAndPublishDiagnostics(request.TextDocument.Uri);
		return Unit.Task;
	}

	public Task<Unit> Handle(DidCloseTextDocumentParams request,
		CancellationToken cancellationToken) =>
		Unit.Task; //ncrunch: no coverage

	public Task<Unit>
		Handle(DidSaveTextDocumentParams request, CancellationToken cancellationToken)
	{
		ParseUpdatedCodeAndPublishDiagnostics(request.TextDocument.Uri);
		return Unit.Task;
	}

	public TextDocumentChangeRegistrationOptions GetRegistrationOptions(
		TextSynchronizationCapability capability, ClientCapabilities clientCapabilities) =>
		new() //ncrunch: no coverage
		{
			DocumentSelector = BaseSelectors.StrictDocumentSelector,
			SyncKind = TextDocumentSyncKind.Incremental
		};

	TextDocumentSaveRegistrationOptions
		IRegistration<TextDocumentSaveRegistrationOptions, TextSynchronizationCapability>.
		GetRegistrationOptions(TextSynchronizationCapability capability,
			ClientCapabilities clientCapabilities) =>
		new() { DocumentSelector = BaseSelectors.StrictDocumentSelector, IncludeText = true }; //ncrunch: no coverage

	TextDocumentOpenRegistrationOptions
		IRegistration<TextDocumentOpenRegistrationOptions, TextSynchronizationCapability>.
		GetRegistrationOptions(TextSynchronizationCapability capability,
			ClientCapabilities clientCapabilities) =>
		new() { DocumentSelector = BaseSelectors.StrictDocumentSelector }; //ncrunch: no coverage

	TextDocumentCloseRegistrationOptions
		IRegistration<TextDocumentCloseRegistrationOptions, TextSynchronizationCapability>.
		GetRegistrationOptions(TextSynchronizationCapability capability,
			ClientCapabilities clientCapabilities) =>
		new() { DocumentSelector = BaseSelectors.StrictDocumentSelector }; //ncrunch: no coverage
}