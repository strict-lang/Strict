using System.Collections.Concurrent;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;

namespace Strict.LanguageServer;

public sealed class StrictDocument
{
	private readonly ConcurrentDictionary<DocumentUri, string[]> strictDocuments = new();
	private List<string> content = new();

	public void Update(DocumentUri uri, IEnumerable<TextDocumentContentChangeEvent> changes)
	{
		content = strictDocuments[uri].ToList();
		foreach (var change in changes)
			UpdateDocument(change);
		strictDocuments[uri] = content.ToArray();
	}

	private void UpdateDocument(TextDocumentContentChangeEvent change)
	{
		if (change.Range != null && content.Count - 1 < change.Range.Start.Line)
			content.Add(change.Text);
		else if (change.Range != null && change.Text == "" &&
			change.Range.Start.Line < change.Range.End.Line)
			HandleForMultiLineDeletion(change.Range.End.Line, change.Range.Start.Line,
				change.Range.Start.Character);
		else
			HandleForDocumentChange(change);
	}

	private void HandleForDocumentChange(TextDocumentContentChangeEvent change)
	{
		if (change.Range != null)
		{
			if (change.Text.Contains('\n'))
				content.Add(change.Text.Split('\n')[^1]);
			else
				content[change.Range.Start.Line] = content[change.Range.Start.Line].
					Remove(change.Range.Start.Character,
						change.Range.End.Character - change.Range.Start.Character).
					Insert(change.Range.Start.Character, change.Text);
		}
	}

	private void HandleForMultiLineDeletion(int endLine, int startLine, int startCharacter)
	{
		content.RemoveRange(endLine, endLine - startLine);
		content[^1] = content[^1][..startCharacter];
	}

	public bool Contains(DocumentUri uri) => strictDocuments.ContainsKey(uri);

	public void AddOrUpdate(DocumentUri uri, params string[] lines) =>
		strictDocuments.AddOrUpdate(uri, lines, (_, _) => lines.ToArray());

	public string[] Get(DocumentUri uri) =>
		strictDocuments.TryGetValue(uri, out var code)
			? code
			: new[] { "" };
}