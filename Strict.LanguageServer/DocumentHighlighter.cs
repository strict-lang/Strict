using OmniSharp.Extensions.LanguageServer.Protocol.Client.Capabilities;
using OmniSharp.Extensions.LanguageServer.Protocol.Document;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using Range = OmniSharp.Extensions.LanguageServer.Protocol.Models.Range;

namespace Strict.LanguageServer;

//ncrunch: no coverage start
public class DocumentHighlighter : IDocumentHighlightHandler
{
	private readonly TextDocumentSynchronizer documentSynchronizer;

	public DocumentHighlighter(TextDocumentSynchronizer documentSynchronizer) =>
		this.documentSynchronizer = documentSynchronizer;

	public async Task<DocumentHighlightContainer?> Handle(DocumentHighlightParams request, CancellationToken cancellationToken)
	{
		await Task.Yield();
		if (!documentSynchronizer.Document.Contains(request.TextDocument.Uri))
			return null;
		var document = documentSynchronizer.Document.Get(request.TextDocument.Uri);
		return new[]
		{
			new DocumentHighlight { Kind = DocumentHighlightKind.Write, Range = new Range(1, 0, Array.IndexOf(document, document[^1]), document[^1].Length) }
		};
	}

	public DocumentHighlightRegistrationOptions GetRegistrationOptions(
		DocumentHighlightCapability capability, ClientCapabilities clientCapabilities) =>
		new() { DocumentSelector = BaseSelectors.StrictDocumentSelector };
}