using System.Text;
using Strict.Bytecode.Instructions;

namespace Strict.Compiler.Assembly;

public sealed partial class InstructionsToLlvmIr
{
	private static void EmitPrint(PrintInstruction print, List<string> lines, EmitContext context)
	{
		var printKey = BuildPrintKey(print, context.Platform);
		var stringLabel = "@str." + BuildPrintLabel(printKey);
		var strGep = context.NextTemp();
		lines.Add($"  {strGep} = getelementptr inbounds [0 x i8], ptr {stringLabel}, i64 0, i64 0");
		if (context.Platform == Platform.Windows)
		{
			var stdoutHandle = context.NextTemp();
			lines.Add($"  {stdoutHandle} = call ptr @GetStdHandle(i32 -11)");
			if (print.ValueRegister.HasValue && !print.ValueIsText)
			{
				var prefixLength = Encoding.UTF8.GetByteCount(print.TextPrefix);
				if (prefixLength > 0)
				{
					var writtenPrefix = context.NextTemp();
					lines.Add($"  {writtenPrefix} = alloca i32");
					lines.Add($"  call i32 @WriteFile(ptr {
						stdoutHandle
					}, ptr {
						strGep
					}, i32 {
						prefixLength
					}, ptr {
						writtenPrefix
					}, ptr null)");
				}
				var numValue = GetRegisterValue(print.ValueRegister.Value, context);
				lines.Add($"  call void @print_number_from_double(ptr {stdoutHandle}, double {numValue})");
			}
			else
			{ //ncrunch: no coverage start
				var textLength = Encoding.UTF8.GetByteCount(print.TextPrefix) + 1;
				var writtenText = context.NextTemp();
				lines.Add($"  {writtenText} = alloca i32");
				lines.Add($"  call i32 @WriteFile(ptr {
					stdoutHandle
				}, ptr {
					strGep
				}, i32 {
					textLength
				}, ptr {
					writtenText
				}, ptr null)");
			} //ncrunch: no coverage end
			return;
		}
		if (print.ValueRegister.HasValue && !print.ValueIsText)
		{
			var numValue = GetRegisterValue(print.ValueRegister.Value, context);
			var bufPtr = context.NextTemp();
			lines.Add($"  {bufPtr} = alloca [64 x i8]");
			var castPtr = context.NextTemp();
			lines.Add($"  {castPtr} = getelementptr [64 x i8], ptr {bufPtr}, i64 0, i64 0");
			var snprintfResult = context.NextTemp();
			lines.Add($"  {
				snprintfResult
			} = call i32 (ptr, i64, ptr, ...) @snprintf(ptr {
				castPtr
			}, i64 64, ptr {
				strGep
			}, double {
				numValue
			})");
			var safeFmt = context.NextTemp();
			lines.Add($"  {safeFmt} = call i32 (ptr, ...) @printf(ptr @str.safe_s, ptr {castPtr})");
		}
		else
		{
			var result = context.NextTemp();
			lines.Add($"  {result} = call i32 (ptr, ...) @printf(ptr {strGep})");
		}
	}

	private static string BuildPrintKey(PrintInstruction print, Platform platform) =>
		platform == Platform.Windows && print.ValueRegister.HasValue && !print.ValueIsText
			? print.TextPrefix
			: print.ValueRegister.HasValue && !print.ValueIsText
				? print.TextPrefix + "%g"
				: print.TextPrefix;

	private static List<(string Label, string Text)> CollectPrintStrings(
		IEnumerable<Instruction> instructions, Platform platform)
	{
		var strings = new List<(string, string)>();
		var seen = new HashSet<string>(StringComparer.Ordinal);
		foreach (var print in instructions.OfType<PrintInstruction>())
		{
			var key = BuildPrintKey(print, platform);
			if (seen.Add(key))
				strings.Add(("str." + BuildPrintLabel(key), key + "\n\0"));
		}
		return strings;
	}

	private static string BuildWindowsPrintNumberHelper() =>
		string.Join("\n", "define void @print_number_from_double(ptr %stdout, double %value) {",
			"entry:", "  %buffer = alloca [64 x i8]",
			"  %bufferStart = getelementptr [64 x i8], ptr %buffer, i64 0, i64 0",
			"  %remainingPtr = alloca i64", "  %writeIndexPtr = alloca i64", "  %writtenPtr = alloca i32",
			"  %number = fptosi double %value to i64", "  %isNegative = icmp slt i64 %number, 0",
			"  %negated = sub i64 0, %number",
			"  %absolute = select i1 %isNegative, i64 %negated, i64 %number",
			"  store i64 %absolute, ptr %remainingPtr", "  store i64 62, ptr %writeIndexPtr",
			"  %newlinePtr = getelementptr i8, ptr %bufferStart, i64 62",
			"  store i8 10, ptr %newlinePtr", "  %isZero = icmp eq i64 %absolute, 0",
			"  br i1 %isZero, label %storeZero, label %digitLoop", "storeZero:",
			"  %zeroIndex = load i64, ptr %writeIndexPtr", "  %zeroStoreIndex = sub i64 %zeroIndex, 1",
			"  store i64 %zeroStoreIndex, ptr %writeIndexPtr",
			"  %zeroPtr = getelementptr i8, ptr %bufferStart, i64 %zeroStoreIndex",
			"  store i8 48, ptr %zeroPtr", "  br label %afterDigits", "digitLoop:",
			"  %current = load i64, ptr %remainingPtr", "  %remainder = urem i64 %current, 10",
			"  %quotient = udiv i64 %current, 10", "  store i64 %quotient, ptr %remainingPtr",
			"  %digitValue = add i64 %remainder, 48", "  %digitByte = trunc i64 %digitValue to i8",
			"  %loopIndex = load i64, ptr %writeIndexPtr", "  %digitStoreIndex = sub i64 %loopIndex, 1",
			"  store i64 %digitStoreIndex, ptr %writeIndexPtr",
			"  %digitPtr = getelementptr i8, ptr %bufferStart, i64 %digitStoreIndex",
			"  store i8 %digitByte, ptr %digitPtr", "  %hasMoreDigits = icmp ne i64 %quotient, 0",
			"  br i1 %hasMoreDigits, label %digitLoop, label %afterDigits", "afterDigits:",
			"  br i1 %isNegative, label %storeSign, label %prepareWrite", "storeSign:",
			"  %signIndex = load i64, ptr %writeIndexPtr", "  %signStoreIndex = sub i64 %signIndex, 1",
			"  store i64 %signStoreIndex, ptr %writeIndexPtr",
			"  %signPtr = getelementptr i8, ptr %bufferStart, i64 %signStoreIndex",
			"  store i8 45, ptr %signPtr", "  br label %prepareWrite", "prepareWrite:",
			"  %startIndex = load i64, ptr %writeIndexPtr",
			"  %outputPtr = getelementptr i8, ptr %bufferStart, i64 %startIndex",
			"  %length64 = sub i64 63, %startIndex", "  %length32 = trunc i64 %length64 to i32",
			"  call i32 @WriteFile(ptr %stdout, ptr %outputPtr, i32 %length32, ptr %writtenPtr, ptr null)",
			"  ret void", "}");

	private static string BuildPrintLabel(string text)
	{
		var result = new StringBuilder(text.Length * 2);
		foreach (var character in text)
			if (char.IsLetterOrDigit(character))
				result.Append(character);
			else
				result.Append('_').Append(((int)character).ToString("X4"));
		return result.ToString();
	}

	private static string BuildStringConstants(List<(string Label, string Text)> strings)
	{
		var lines = new List<string>();
		foreach (var (label, text) in strings)
		{
			var escaped = EscapeForLlvm(text);
			var length = CountLlvmStringBytes(escaped);
			lines.Add($"@{label} = private unnamed_addr constant [{length} x i8] c\"{escaped}\"");
		}
		return string.Join("\n", lines);
	}

	private static string EscapeForLlvm(string text)
	{
		var result = new StringBuilder();
		foreach (var character in text)
			if (character == '\n')
				result.Append("\\0A");
			else if (character == '\0')
				result.Append("\\00");
			else if (character == '\\')
				result.Append("\\5C"); //ncrunch: no coverage
			else if (character == '"')
				result.Append("\\22"); //ncrunch: no coverage
			else if (character is >= ' ' and <= '~')
				result.Append(character);
			else
				result.Append($"\\{(int)character:X2}"); //ncrunch: no coverage
		return result.ToString();
	}

	private static int CountLlvmStringBytes(string escaped)
	{
		var count = 0;
		for (var index = 0; index < escaped.Length; index++)
		{
			count++;
			if (escaped[index] == '\\' && index + 2 < escaped.Length)
				index += 2;
		}
		return count;
	}
}
