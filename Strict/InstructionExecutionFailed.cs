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
    string sourceFilePath = "", Exception? inner = null)
    : base(BuildMessage(message, instructions, failingIndex, methodContext, sourceLines,
      sourceFilePath), inner) { }

  private static string BuildMessage(string message, IReadOnlyList<Instruction> instructions,
    int failingIndex, string methodContext, string[]? sourceLines, string sourceFilePath)
  {
    var builder = new StringBuilder();
    builder.Append(message);
    AppendInstructionWindow(builder, instructions, failingIndex, methodContext);
    AppendSourceSection(builder, instructions, failingIndex, methodContext, sourceLines,
      sourceFilePath);
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
