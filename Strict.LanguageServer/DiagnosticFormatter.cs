using System.Text.RegularExpressions;
using OmniSharp.Extensions.LanguageServer.Protocol.Models;
using Strict.Language;
using Range = OmniSharp.Extensions.LanguageServer.Protocol.Models.Range;

namespace Strict.LanguageServer;

public static class DiagnosticFormatter
{
	private static readonly Regex LineNumberRegex = new(@":line (\d+)", RegexOptions.Compiled);

	public static Diagnostic FromException(Exception exception, IReadOnlyList<string> content)
	{
		var code = exception.GetType().Name;
		var lineNumber = GetLineNumber(exception, content.Count);
		return new Diagnostic
		{
			Code = code,
			Severity = DiagnosticSeverity.Error,
			Message = FormatMessage(code, exception.Message),
			Range = GetErrorTextRange(content, lineNumber),
			Source = "strict"
		};
	}

	public static string FormatMessage(string errorCode, string exceptionMessage)
	{
		var humanized = HumanizePascalCase(errorCode);
		var detail = ExtractDetail(exceptionMessage);
		return detail.Length == 0
			? humanized
			: humanized + ": " + detail;
	}

	public static string HumanizePascalCase(string name)
	{
		if (string.IsNullOrEmpty(name))
			return name;
		var words = new List<string>();
		var start = 0;
		for (var i = 1; i < name.Length; i++)
		{
			var previous = name[i - 1];
			var current = name[i];
			var splitBeforeCurrent =
				char.IsUpper(current) && (char.IsLower(previous) ||
					i + 1 < name.Length && char.IsLower(name[i + 1]) && char.IsUpper(previous));
			if (!splitBeforeCurrent)
				continue;
			words.Add(name[start..i]);
			start = i;
		}
		words.Add(name[start..]);
		return string.Join(' ', words.Select((word, index) =>
		{
			var lower = word.ToLowerInvariant();
			return index == 0
				? char.ToUpperInvariant(lower[0]) + lower[1..]
				: lower;
		}));
	}

	public static string ExtractDetail(string exceptionMessage)
	{
		if (string.IsNullOrWhiteSpace(exceptionMessage))
			return "";
		var atIndex = exceptionMessage.IndexOf("\n   at ", StringComparison.Ordinal);
		var detail = (atIndex >= 0
			? exceptionMessage[..atIndex]
			: exceptionMessage).Trim();
		// Drop pure path/location lines that only repeat file context the editor already shows
		if (detail.StartsWith("at ", StringComparison.Ordinal) ||
			detail.Contains(":line ", StringComparison.Ordinal))
			return "";
		return detail;
	}

	public static int GetLineNumber(Exception exception, int lineCount)
	{
		if (exception is ParsingFailed parsingFailed)
			return int.Clamp(parsingFailed.FileLineNumber, 0, Math.Max(0, lineCount - 1));
		var match = LineNumberRegex.Match(exception.Message);
		if (match.Success && int.TryParse(match.Groups[1].Value, out var oneBased))
			return int.Clamp(oneBased - 1, 0, Math.Max(0, lineCount - 1));
		return 0;
	}

	public static Range GetErrorTextRange(IReadOnlyList<string> content, int lineNumber)
	{
		if (content.Count == 0)
			return new Range(0, 0, 0, 1);
		lineNumber = int.Clamp(lineNumber, 0, content.Count - 1);
		var line = content[lineNumber];
		// Empty or whitespace-only: mark the whole line so the editor shows a clear squiggle
		if (line.Length == 0)
			return lineNumber + 1 < content.Count
				? new Range(lineNumber, 0, lineNumber + 1, 0)
				: new Range(lineNumber, 0, lineNumber, 1);
		var start = 0;
		while (start < line.Length && char.IsWhiteSpace(line[start]))
			start++;
		if (start >= line.Length)
			return new Range(lineNumber, 0, lineNumber, line.Length);
		return new Range(lineNumber, start, lineNumber, line.Length);
	}
}
