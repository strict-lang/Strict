using System.Collections.Concurrent;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;

namespace Strict.LanguageServer;

// ReSharper disable once HollowTypeName
public sealed class StrictDocumentManager
{
	private readonly ConcurrentDictionary<DocumentUri, string[]> strictDocuments = new();

	public void Update(DocumentUri uri, IEnumerable<TextDocumentContentChangeEvent> changes)
	{
		var content = strictDocuments[uri].ToList();
		foreach (var change in changes)
			if (change.Range != null)
			{
				if (content.Count - 1 < change.Range.Start.Line)
					content.Add(change.Text);
				else
					content[change.Range.Start.Line] = content[change.Range.Start.Line].
						Remove(change.Range.Start.Character,
							change.Range.End.Character - change.Range.Start.Character).
						Insert(change.Range.Start.Character, change.Text);
			}
		strictDocuments[uri] = content.ToArray();
	}

	public bool Contains(DocumentUri uri) => strictDocuments.ContainsKey(uri);

	public void AddOrUpdate(DocumentUri uri, params string[] lines) =>
		strictDocuments.AddOrUpdate(uri, lines, (_, _) => lines);

	public string[] Get(DocumentUri uri) =>
		strictDocuments.TryGetValue(uri, out var content)
			? content
			: new[] { "" };
}