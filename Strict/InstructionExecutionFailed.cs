using System.Text;
using Strict.Bytecode;
using Strict.Bytecode.Instructions;

namespace Strict;

/// <summary>
/// Thrown when the VirtualMachine fails to execute an instruction. Provides rich context:
/// the failing instruction with a surrounding window, source lines with line numbers from the
/// .strict file, and a clickable stack trace entry (same format as ParsingFailed).
/// </summary>
public sealed class InstructionExecutionFailed : Exception
{
	public InstructionExecutionFailed(string message, IReadOnlyList<Instruction> instructions,
		int failingIndex, string methodContext, string[]? sourceLines = null,
		string sourceFilePath = "", List<(string method, string filePath, int sourceLine)>? callStack = null,
		Exception? inner = null)
		: base(BuildMessage(message, instructions, failingIndex, methodContext, sourceLines,
			sourceFilePath, callStack), inner) { }

	private static string BuildMessage(string message, IReadOnlyList<Instruction> instructions,
		int failingIndex, string methodContext, string[]? sourceLines, string sourceFilePath,
		List<(string method, string filePath, int sourceLine)>? callStack)
	{
		var builder = new StringBuilder();
		builder.Append(message);
		AppendInstructionWindow(builder, instructions, failingIndex, methodContext);
		AppendSourceSection(builder, instructions, failingIndex, methodContext, sourceLines,
			sourceFilePath);
		if (callStack?.Count > 0)
			AppendCallStack(builder, callStack);
		//builder.Append("\nInstructions so far with SourceLine: "+BinaryExecutable.hasSourceLines+", without SourceLines: "+BinaryExecutable.hasNoSourceLines);
		return builder.ToString().TrimEnd();
	}

	private static void AppendInstructionWindow(StringBuilder builder,
		IReadOnlyList<Instruction> instructions, int failingIndex, string methodContext)
	{
		if (!string.IsNullOrEmpty(methodContext))
		{
			builder.AppendLine();
			builder.Append("   in ");
			builder.Append(methodContext);
		}
		if (instructions.Count == 0)
			return;
		builder.AppendLine();
		builder.Append("   Instructions (");
		builder.Append(failingIndex);
		builder.Append('/');
		builder.Append(instructions.Count - 1);
		builder.AppendLine("):");
		var start = Math.Max(0, failingIndex - ContextWindowSize);
		var end = Math.Min(instructions.Count - 1, failingIndex + ContextWindowSize);
		for (var index = start; index <= end; index++)
		{
			var instruction = instructions[index];
			var marker = index == failingIndex ? ">>>" : "   ";
			builder.Append("   ");
			builder.Append(marker);
			builder.Append(' ');
			builder.Append(index.ToString().PadLeft(4));
			builder.Append(": ");
			builder.Append(instruction);
			if (instruction.SourceLine != 0)
			{
				builder.Append("  (:line ");
				builder.Append(instruction.SourceLine + 1);
				builder.Append(')');
			}
			builder.AppendLine();
		}
	}

	private const int ContextWindowSize = 5;

	private static void AppendSourceSection(StringBuilder builder,
		IReadOnlyList<Instruction> instructions, int failingIndex, string methodContext,
		string[]? sourceLines, string sourceFilePath)
	{
		if (sourceLines == null || sourceLines.Length == 0)
			return;
		var (minLine, maxLine) = GetSourceLineRange(instructions,
			Math.Max(0, failingIndex - ContextWindowSize),
			Math.Min(instructions.Count - 1, failingIndex + ContextWindowSize));
		if (minLine < 0)
			return;
		var failingSourceLine = instructions[failingIndex].SourceLine != 0
			? instructions[failingIndex].SourceLine
			: minLine;
		var methodName = GetMethodName(methodContext);
		var methodStartLine = FindMethodStartLine(sourceLines, methodName);
		builder.AppendLine();
		builder.Append("   ");
		builder.AppendLine(Path.GetFileName(sourceFilePath));
		var sourceStart = methodStartLine >= 0 ? methodStartLine : Math.Max(0, minLine - 1);
		var sourceEnd = Math.Min(sourceLines.Length - 1, maxLine + 1);
		for (var lineIndex = sourceStart; lineIndex <= sourceEnd; lineIndex++)
		{
			builder.Append(lineIndex == failingSourceLine ? ">>>" : "   ");
			builder.Append(lineIndex + 1);
			builder.Append(": ");
			builder.AppendLine(sourceLines[lineIndex].Replace("\t", "  "));
		}
		builder.AppendLine();
		builder.Append("   at ");
		builder.Append(methodContext);
		builder.Append(" in ");
		builder.Append(sourceFilePath);
		builder.Append(":line ");
		builder.Append(failingSourceLine + 1);
	}

	private static void AppendCallStack(StringBuilder builder,
		List<(string method, string filePath, int sourceLine)> callStack)
	{
		builder.AppendLine();
		for (var i = 0; i < callStack.Count; i++)
		{
			var (method, filePath, sourceLine) = callStack[i];
			builder.Append("   at ");
			var displayMethod = method;
			if (!string.IsNullOrEmpty(filePath) && !method.Contains('/'))
			{
				var fileTypeInfo = ExtractTypeInfoFromFilePath(filePath);
				if (!string.IsNullOrEmpty(fileTypeInfo) && method.EndsWith(Path.GetFileName(fileTypeInfo).Split('.')[0] + "." + GetMethodName(method)))
					displayMethod = fileTypeInfo + "." + GetMethodName(method);
			}
			builder.Append(displayMethod);
			if (!string.IsNullOrEmpty(filePath))
			{
				builder.Append(" in ");
				builder.Append(filePath);
				if (sourceLine >= 0)
				{
					builder.Append(":line ");
					builder.Append(sourceLine + 1);
				}
			}
			builder.AppendLine();
		}
	}

	private static string ExtractTypeInfoFromFilePath(string filePath)
	{
		if (string.IsNullOrEmpty(filePath))
			return "";
		var fileName = Path.GetFileNameWithoutExtension(filePath);
		var directoryName = Path.GetDirectoryName(filePath) ?? "";
		var parts = directoryName.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
		if (parts.Length > 0)
		{
			var lastFolder = parts[^1];
			return lastFolder + "/" + fileName;
		}
		return fileName;
	}

	private static string GetMethodName(string methodContext)
	{
		var dotIndex = methodContext.LastIndexOf('.');
		return dotIndex >= 0 ? methodContext[(dotIndex + 1)..] : methodContext;
	}

	private static int FindMethodStartLine(string[] sourceLines, string methodName)
	{
		for (var lineIndex = 0; lineIndex < sourceLines.Length; lineIndex++)
		{
			var line = sourceLines[lineIndex];
			if (line.StartsWith('\t'))
				continue;
			if (line.StartsWith(methodName, StringComparison.Ordinal) ||
				line.Contains('(' + methodName + ')', StringComparison.Ordinal))
				return lineIndex;
		}
		return -1;
	}

	private static (int min, int max) GetSourceLineRange(IReadOnlyList<Instruction> instructions,
		int startIndex, int endIndex)
	{
		var min = int.MaxValue;
		var max = int.MinValue;
		for (var index = startIndex; index <= endIndex; index++)
		{
			if (instructions[index].SourceLine == 0)
				continue;
			var sourceLine = instructions[index].SourceLine;
			if (sourceLine < min)
				min = sourceLine;
			if (sourceLine > max)
				max = sourceLine;
		}
		return min == int.MaxValue ? (-1, -1) : (min, max);
	}
}
