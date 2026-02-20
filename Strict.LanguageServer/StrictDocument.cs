using System.Collections.Concurrent;
using System.Collections.Immutable;
using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using OmniSharp.Extensions.LanguageServer.Protocol.Server;
using OmniSharp.Extensions.LanguageServer.Protocol.Window;
using Strict.Language;
using Strict.Runtime;
using Range = OmniSharp.Extensions.LanguageServer.Protocol.Models.Range;

namespace Strict.LanguageServer;

public sealed class StrictDocument
{
	private readonly ConcurrentDictionary<DocumentUri, string[]> strictDocuments = new();
	private List<string> content = [];
	private readonly BytecodeInterpreter vm = new();

	public void Update(DocumentUri uri, TextDocumentContentChangeEvent[] changes)
	{
		content = strictDocuments[uri].ToList();
		if (changes.Length > 0)
			UpdateDocument(changes[0]);
		strictDocuments[uri] = content.ToArray();
	}

	private void UpdateDocument(TextDocumentContentChangeEvent change)
	{
		if (change.Range is not null &&
			change.Text.StartsWith(Environment.NewLine, StringComparison.Ordinal) &&
			change.Range.Start.Line == change.Range.End.Line)
			content.Insert(change.Range.Start.Line + 1, change.Text[2..]);
		else if (change.Range is not null && content.Count - 1 < change.Range.Start.Line)
			AddSingleOrMultiLineNewText(change);
		else if (change.Range is not null && change.Range.Start.Line < change.Range.End.Line)
		{
			HandleForMultiLineDeletion(change.Range.Start, change.Range.End);
			if (change.Text is not "")
				content[change.Range.Start.Line] = content[change.Range.Start.Line]. //ncrunch: no coverage
					Insert(change.Range.Start.Character, change.Text);
		}
		else
			HandleForDocumentChange(change);
	}

	private void AddSingleOrMultiLineNewText(TextDocumentContentChangeEvent change)
	{
		if (change.Text.Contains('\n'))
			content.AddRange(change.Text.Split('\n'));
		else
			content.Add(change.Text);
	}

	private void HandleForDocumentChange(TextDocumentContentChangeEvent change)
	{
		if (change.Range is null)
			return; //ncrunch: no coverage
		if (change.Text.Contains('\n'))
			content.Add(change.Text.Split('\n')[^1]); //ncrunch: no coverage
		else if (change.Range.End.Character - change.Range.Start.Character > 0)
			content[change.Range.Start.Line] = content[change.Range.Start.Line].
				Remove(change.Range.Start.Character,
					change.Range.End.Character - change.Range.Start.Character).
				Insert(change.Range.Start.Character, change.Text);
		else
			content[change.Range.Start.Line] = content[change.Range.Start.Line].
				Insert(change.Range.Start.Character, change.Text);
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
		content[start.Line] =
			content[start.Line][..start.Character] + content[end.Line][end.Character..];
		content.RemoveRange(start.Line + 1, end.Line - start.Line);
	}

	private void RemoveLinesTillEndAndUpdateStartLine(Position start, Position end)
	{
		content.RemoveRange(start.Character is 0
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
			: [""];

	public IEnumerable<Diagnostic> GetDiagnostics(Package package, DocumentUri uri,
		ILanguageServerFacade languageServer)
	{
		var diagnostics = ImmutableArray<Diagnostic>.Empty.ToBuilder();
		try
		{
			ParseCurrentFile(package, uri, languageServer);
		}
		catch (Exception exception)
		{
			languageServer.Window.LogError(exception.Message);
			diagnostics.Add(new Diagnostic
			{
				Code = exception.GetType().Name,
				Severity = DiagnosticSeverity.Error,
				Message = exception.GetType().Name + ": " + exception.Message,
				Range = GetErrorTextRange(exception.Message),
				Source = exception.Source,
				Tags = new Container<DiagnosticTag>(DiagnosticTag.Unnecessary)
			});
		}
		return diagnostics;
	}

	private void ParseCurrentFile(Package package, DocumentUri uri, ILanguageServerFacade languageServer)
	{
		var folderName = uri.Path.GetFolderName();
		var subPackage = package.Find(folderName) ?? new Package(package, folderName);
		var type = subPackage.SynchronizeAndGetType(uri.Path.GetFileName(), content);
		if (type is { IsTrait: false })
		{
			var methods = ParseTypeMethods(type.Methods);
			if (methods != null)
				// @formatter:off
				new RunnerService()
					.AddService(new TestRunner(languageServer,methods))
					.AddService(new VariableValueEvaluator(languageServer, Get(uri)))
					.RunAllServices();
			// @formatter:on
		}
	}

	private static IEnumerable<Method>? ParseTypeMethods(IEnumerable<Method> methods)
	{
		foreach (var method in methods.Where(method => !method.IsGeneric))
			if (method.GetBodyAndParseIfNeeded() is Body body)
				yield return body.Method; //ncrunch: no coverage
	}

	private Range GetErrorTextRange(string errorMessage)
	{
		int.TryParse(errorMessage.Split(' ')[^1], out var lineNumber);
		lineNumber = int.Clamp(lineNumber - 1, 0, content.Count - 1);
		return new Range(lineNumber, content[lineNumber].TrimStart().Length, lineNumber,
			content[lineNumber].Length);
	}

	public void InitializeContent(DocumentUri uri) => content = strictDocuments[uri].ToList();
}