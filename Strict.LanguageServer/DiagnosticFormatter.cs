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
			Message = FormatMessage(code, BuildExceptionText(exception)),
			Range = GetErrorTextRange(content, lineNumber),
			Source = "strict"
		};
	}

	public static string BuildExceptionText(Exception exception)
	{
		var text = exception.Message ?? "";
		var extra = FormatExtraInfo(exception);
		return extra.Length == 0
			? text
			: text + "\n" + extra;
	}

	public static string FormatExtraInfo(Exception exception)
	{
		var parts = new List<string>();
		var type = exception.GetType();
		if (!string.IsNullOrEmpty(type.Namespace))
			parts.Add(type.Namespace + "." + type.Name);
		if (exception.InnerException != null)
			parts.Add("Caused by " + exception.InnerException.GetType().Name + ": " +
				FirstLine(exception.InnerException.Message));
		return string.Join('\n', parts);
	}

	public static string FormatMessage(string errorCode, string exceptionMessage)
	{
		var detail = ExtractDetail(exceptionMessage);
var humanized = HumanizePascalCase(errorCode);
		if (errorCode is "InterpreterExecutionFailed" or "TestFailed")
			return detail.Length == 0 ? humanized : detail;
		if (detail.Length == 0)
			return humanized;
		if (detail.StartsWith(humanized + ":", StringComparison.OrdinalIgnoreCase))
			return detail;
		return humanized + ": " + detail;
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
		var withoutStack = (atIndex >= 0
			? exceptionMessage[..atIndex]
			: exceptionMessage).Trim();
		var useful = new List<string>();
foreach (var rawLine in withoutStack.Split('\n'))
		{
			var line = rawLine.Trim().TrimEnd('\r');
			if (line.Length == 0)
				continue;
			if (line.StartsWith("at ", StringComparison.Ordinal) &&
				line.Contains(":line ", StringComparison.Ordinal))
			{
				if (useful.Count == 0)
					return "";
				continue;
			}
			if (line.Contains(":line ", StringComparison.Ordinal))
				break;
			if (line.StartsWith("Instructions ", StringComparison.Ordinal) ||
				line.StartsWith(">>>", StringComparison.Ordinal) ||
				line.Length > 0 && char.IsDigit(line[0]) && line.Contains(':'))
				break;
			var cleaned = StripTypePrefix(line);
			if (cleaned.Length > 0)
				useful.Add(cleaned);
		}
		return string.Join('\n', useful);
	}

	private static string StripTypePrefix(string text)
	{
		var colon = text.IndexOf(':');
		if (colon <= 0 || text[..colon].Contains(' '))
			return text;
		var after = text[(colon + 1)..].Trim();
		return after.Length > 0 && !after.StartsWith("at ", StringComparison.Ordinal)
			? after
			: "";
	}

	private static string FirstLine(string? text)
	{
		if (string.IsNullOrEmpty(text))
			return "";
		var newline = text.IndexOf('\n');
		return (newline < 0
			? text
			: text[..newline]).Trim();
	}

	public static int GetLineNumber(Exception exception, int lineCount)
	{
		if (exception is ParsingFailed parsingFailed)
			return int.Clamp(parsingFailed.FileLineNumber, 0, Math.Max(0, lineCount - 1));
		var matches = LineNumberRegex.Matches(exception.Message);
		if (matches.Count > 0 && int.TryParse(matches[^1].Groups[1].Value, out var oneBased))
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
