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
			HandleForMultiLineDeletion(change.Range.Start, change.Range.End);
		else
			HandleForDocumentChange(change);
	}

	private void HandleForDocumentChange(TextDocumentContentChangeEvent change)
	{
		if (change.Range != null)
		{
			if (change.Text.Contains('\n'))
				content.Add(change.Text.Split('\n')[^1]);
			else if (change.Range.End.Character - change.Range.Start.Character > 0)
				content[change.Range.Start.Line] = content[change.Range.Start.Line].
					Remove(change.Range.Start.Character,
						change.Range.End.Character - change.Range.Start.Character).
					Insert(change.Range.Start.Character, change.Text);
			else
				content[change.Range.Start.Line] = content[change.Range.Start.Line].
					Insert(change.Range.Start.Character, change.Text);
		}
	}

	private void HandleForMultiLineDeletion(Position start, Position end)
	{
		if (end.Line == content.Count - 1 && end.Character >= content[end.Line].Length)
			RemoveLinesTillEndAndUpdateStartLine(start, end);
		else
			RemoveLinesInMiddleAndUpdateStartAndEndLines(start, end);
	}

	private void RemoveLinesInMiddleAndUpdateStartAndEndLines(Position start, Position end)
	{
		content[start.Line] = content[start.Line][..start.Character];
		content[end.Line] = content[end.Line][end.Character..];
		if (end.Line - start.Line > 1)
			content.RemoveRange(start.Line + 1, end.Line - (start.Line + 1));
	}

	private void RemoveLinesTillEndAndUpdateStartLine(Position start, Position end)
	{
		content.RemoveRange(start.Character == 0
			? start.Line
			: start.Line + 1, end.Line - start.Line);
		content[^1] = content[^1][..start.Character];
	}

	public bool Contains(DocumentUri uri) => strictDocuments.ContainsKey(uri);

	public void AddOrUpdate(DocumentUri uri, params string[] lines) =>
		strictDocuments.AddOrUpdate(uri, lines, (_, _) => lines.ToArray());

	public string[] Get(DocumentUri uri) =>
		strictDocuments.TryGetValue(uri, out var code)
			? code
			: new[] { "" };

	public IEnumerable<Diagnostic> GetDiagnostics() =>
		new List<Diagnostic>()
		{
			new()
			{
				Code = "DiagnosticInfo",
				Message = "Key is not complete",
				Severity = DiagnosticSeverity.Warning,
				Range = ((0, 0), (1, 3)),
				Source = "STRICT"
			}
		};
}