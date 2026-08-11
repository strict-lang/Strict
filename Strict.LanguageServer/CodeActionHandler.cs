using OmniSharp.Extensions.LanguageServer.Protocol;
using OmniSharp.Extensions.LanguageServer.Protocol.Client.Capabilities;
using OmniSharp.Extensions.LanguageServer.Protocol.Document;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using Strict.Language;
using Range = OmniSharp.Extensions.LanguageServer.Protocol.Models.Range;

namespace Strict.LanguageServer;

public sealed class CodeActionHandler(StrictDocument document) : ICodeActionHandler
{
	public Task<CommandOrCodeActionContainer?> Handle(CodeActionParams request,
		CancellationToken cancellationToken)
	{
		var actions = new List<CommandOrCodeAction>();
		var lines = document.Get(request.TextDocument.Uri);
		foreach (var diagnostic in request.Context.Diagnostics)
		{
			var code = diagnostic.Code?.ToString();
			if (string.IsNullOrEmpty(code))
				continue;
			var fix = CreateFix(request.TextDocument.Uri, lines, diagnostic, code);
			if (fix != null)
				actions.Add(fix);
		}
		return Task.FromResult<CommandOrCodeActionContainer?>(new CommandOrCodeActionContainer(actions));
	}

	private static CodeAction? CreateFix(DocumentUri uri, string[] lines, Diagnostic diagnostic,
		string code) =>
		code switch
		{
			nameof(TypeParser.EmptyLineIsNotAllowed) => CreateDeleteLineAction(uri, lines, diagnostic,
				"Remove empty line"),
			nameof(TypeParser.ExtraWhitespacesFoundAtEndOfLine) => CreateTrimEndAction(uri, lines,
				diagnostic),
			nameof(TypeParser.ExtraWhitespacesFoundAtBeginningOfLine) => CreateLeadingWhitespaceFix(
				uri, lines, diagnostic),
			_ => null
		};

	private static CodeAction CreateDeleteLineAction(DocumentUri uri, string[] lines,
		Diagnostic diagnostic, string title)
	{
		var line = diagnostic.Range.Start.Line;
		line = int.Clamp(line, 0, Math.Max(0, lines.Length - 1));
		var range = line + 1 < lines.Length
			? new Range(line, 0, line + 1, 0)
			: line > 0
				? new Range(line - 1, lines[line - 1].Length, line, lines[line].Length)
				: new Range(line, 0, line, lines[line].Length);
		return QuickFix(uri, diagnostic, title, range, "");
	}

	private static CodeAction? CreateTrimEndAction(DocumentUri uri, string[] lines,
		Diagnostic diagnostic)
	{
		var line = diagnostic.Range.Start.Line;
		if (line < 0 || line >= lines.Length)
			return null;
		var text = lines[line];
		var trimmed = text.TrimEnd();
		if (trimmed.Length == text.Length)
			return null;
		return QuickFix(uri, diagnostic, "Remove trailing whitespace",
			new Range(line, 0, line, text.Length), trimmed);
	}

	private static CodeAction? CreateLeadingWhitespaceFix(DocumentUri uri, string[] lines,
		Diagnostic diagnostic)
	{
		var line = diagnostic.Range.Start.Line;
		if (line < 0 || line >= lines.Length)
			return null;
		var text = lines[line];
		var index = 0;
		var tabCount = 0;
		while (index < text.Length && (text[index] == '\t' || text[index] == ' '))
		{
			if (text[index] == '\t')
				tabCount++;
			else if (text[index] == ' ')
			{
				// Convert each group of leading spaces into tabs (4 spaces → 1 tab)
				var spaces = 0;
				while (index < text.Length && text[index] == ' ')
				{
					spaces++;
					index++;
				}
				tabCount += Math.Max(1, spaces / 4);
				continue;
			}
			index++;
		}
		var fixedLine = new string('\t', tabCount) + text[index..].TrimEnd();
		if (fixedLine == text)
			return null;
		return QuickFix(uri, diagnostic, "Fix leading whitespace (use tabs)",
			new Range(line, 0, line, text.Length), fixedLine);
	}

	private static CodeAction QuickFix(DocumentUri uri, Diagnostic diagnostic, string title,
		Range range, string newText) =>
		new()
		{
			Title = title,
			Kind = CodeActionKind.QuickFix,
			IsPreferred = true,
			Diagnostics = new Container<Diagnostic>(diagnostic),
			Edit = new WorkspaceEdit
			{
				Changes = new Dictionary<DocumentUri, IEnumerable<TextEdit>>
				{
					[uri] = [new TextEdit { Range = range, NewText = newText }]
				}
			}
		};

	public CodeActionRegistrationOptions GetRegistrationOptions(CodeActionCapability capability,
		ClientCapabilities clientCapabilities) =>
		new()
		{
			DocumentSelector = BaseSelectors.StrictDocumentSelector,
			CodeActionKinds = new Container<CodeActionKind>(CodeActionKind.QuickFix),
			ResolveProvider = false
		};
}
