using OmniSharp.Extensions.LanguageServer.Protocol;
using System.Collections.Concurrent;

namespace Strict.LanguageServer;

// ReSharper disable once HollowTypeName
public sealed class StrictDocumentManager
{
	private readonly ConcurrentDictionary<DocumentUri, string> strictDocuments = new();
	public void Update(DocumentUri uri, string content) => strictDocuments.AddOrUpdate(uri, content, (_, _) => content);

	public string Get(DocumentUri uri) =>
		strictDocuments.TryGetValue(uri, out var content)
			? content
			: string.Empty;
}
