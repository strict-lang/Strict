using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace Strict.Compiler.Assembly;

/// <summary>
/// Compiles an MLIR .mlir file into a native executable via a three-step pipeline:
///   1. mlir-opt: Lower arith+func dialects to LLVM dialect
///   2. mlir-translate: Convert LLVM dialect to LLVM IR (.ll)
///   3. clang: Compile LLVM IR to native executable with -O2
/// Requires mlir-opt, mlir-translate, and clang on PATH.
/// Falls back gracefully when MLIR tools are unavailable — use IsAvailable to check.
/// </summary>
public sealed class MlirLinker : Linker
{
	public override async Task<string> CreateExecutable(string asmFilePath, Platform platform,
		bool hasPrintCalls = false)
	{
		var mlirOptPath = ToolRunner.FindTool("mlir-opt") ??
			throw new ToolNotFoundException("mlir-opt",
				"https://github.com/llvm/llvm-project/releases (install MLIR tools or " +
				"on Windows use Msys2 and 'pacman -S mingw-w64-x86_64-mlir')");
		var mlirTranslatePath = ToolRunner.FindTool("mlir-translate") ??
			throw new ToolNotFoundException("mlir-translate",
				"https://github.com/llvm/llvm-project/releases (install MLIR tools or " +
				"on Windows use Msys2 and 'pacman -S mingw-w64-x86_64-mlir')");
		var clangPath = ToolRunner.FindTool("clang") ??
			throw new ToolNotFoundException("clang", "https://releases.llvm.org");
		var mlirContent = await File.ReadAllTextAsync(asmFilePath);
		var hasGpuOps = mlirContent.Contains("gpu.launch", StringComparison.Ordinal);
		var llvmDialectPath = Path.ChangeExtension(asmFilePath, ".llvm.mlir");
		var optArgs = hasGpuOps
			? BuildMlirOptArgsWithGpu(asmFilePath, llvmDialectPath)
			: BuildMlirOptArgs(asmFilePath, llvmDialectPath);
		ToolRunner.RunProcess(mlirOptPath, optArgs);
		ToolRunner.EnsureOutputFileExists(llvmDialectPath, "mlir-opt", platform);
		var llvmIrPath = Path.ChangeExtension(asmFilePath, ".ll");
		ToolRunner.RunProcess(mlirTranslatePath,
			$"--mlir-to-llvmir \"{llvmDialectPath}\" -o \"{llvmIrPath}\"");
		ToolRunner.EnsureOutputFileExists(llvmIrPath, "mlir-translate", platform);
		if (platform == Platform.Windows && hasPrintCalls)
			await File.WriteAllTextAsync(llvmIrPath,
				RewriteWindowsPrintRuntime(await File.ReadAllTextAsync(llvmIrPath)));
		var exeFilePath = platform == Platform.Windows
			? Path.ChangeExtension(asmFilePath, ".exe")
			: Path.ChangeExtension(asmFilePath, null);
		var arguments = hasGpuOps
			? BuildGpuClangArgs(llvmIrPath, exeFilePath, platform)
			: BuildClangArgs(llvmIrPath, exeFilePath, platform, hasPrintCalls);
		ToolRunner.RunProcess(clangPath, arguments);
		ToolRunner.EnsureOutputFileExists(exeFilePath, "clang", platform);
		return exeFilePath;
	}

	private static string BuildMlirOptArgs(string inputPath, string outputPath) =>
		$"\"{inputPath}\" --canonicalize --cse --symbol-dce --convert-scf-to-cf --convert-arith-to-llvm " +
		$"--convert-func-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts -o \"{outputPath}\"";

	private static string BuildMlirOptArgsWithGpu(string inputPath, string outputPath) =>
		$"\"{inputPath}\" --canonicalize --cse --symbol-dce " +
		"--gpu-kernel-outlining --convert-scf-to-cf " +
		"--convert-gpu-to-nvvm --gpu-to-llvm --convert-nvvm-to-llvm " +
		"--convert-memref-to-llvm --convert-arith-to-llvm " +
		$"--convert-func-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts -o \"{outputPath}\"";

	private static string BuildGpuClangArgs(string inputPath, string outputPath, Platform platform)
	{
		var quotedInputPath = $"\"{inputPath}\"";
		var quotedOutputPath = $"\"{outputPath}\"";
		return platform switch
		{
			Platform.Linux => $"{quotedInputPath} -o {quotedOutputPath} -O2 -lcuda -lcudart",
			Platform.Windows =>
				$"{quotedInputPath} -o {quotedOutputPath} -O2 -lcuda -lcudart -Wno-override-module",
			_ => $"{quotedInputPath} -o {quotedOutputPath} -O2 -lcuda -lcudart -Wno-override-module"
		};
	}
	private static string BuildClangArgs(string inputPath, string outputPath, Platform platform,
		bool hasPrintCalls = false)
	{
		var quotedInputPath = $"\"{inputPath}\"";
		var quotedOutputPath = $"\"{outputPath}\"";
		const string LinuxSizeFlags = "-Oz -s -Wl,--gc-sections -Wl,--strip-all -Wl,--build-id=none -fno-unwind-tables -fno-asynchronous-unwind-tables";
		const string WindowsSizeFlags = "-Oz -nostdlib -lkernel32 -Wl,/ENTRY:main -Wl,/OPT:REF -Wl,/OPT:ICF -Wl,/INCREMENTAL:NO -Wl,/DEBUG:NONE";
		return platform switch
		{
			Platform.Windows =>
				$"{quotedInputPath} -o {quotedOutputPath} {WindowsSizeFlags} -Wno-override-module",
			Platform.Linux when OperatingSystem.IsWindows() =>
				$"{quotedInputPath} -o {quotedOutputPath} -Oz -Wno-override-module",
			//ncrunch: no coverage start
			Platform.Linux => hasPrintCalls
				? $"{quotedInputPath} -o {quotedOutputPath} {LinuxSizeFlags} -Wno-override-module"
				: $"{quotedInputPath} -o {quotedOutputPath} {LinuxSizeFlags} -nostdlib -Wl,-e,main -Wno-override-module",
			Platform.MacOS =>
				$"{quotedInputPath} -o {quotedOutputPath} -Oz -Wl,-dead_strip " +
				$"-Wno-override-module",
			_ => throw new NotSupportedException("Unsupported platform: " + platform)
		}; //ncrunch: no coverage end
	}

	private static string RewriteWindowsPrintRuntime(string llvmIr)
	{
		if (!llvmIr.Contains("@printf(", StringComparison.Ordinal))
			return EnsureWindowsPrintRuntimeSupport(llvmIr); //ncrunch: no coverage
		var stringLengths = ParseStringLengths(llvmIr);
		var replacementIndex = 0;
		var rewritten = PrintWithNumberRegex.Replace(llvmIr, match =>
			BuildNumericPrintReplacement(match, stringLengths, replacementIndex++));
		rewritten = PrintTextRegex.Replace(rewritten, match =>
			BuildTextPrintReplacement(match, stringLengths, replacementIndex++)); //ncrunch: no coverage
		rewritten = rewritten.Replace("declare i32 @printf(ptr, ...)\r\n\r\n", string.Empty,
			StringComparison.Ordinal);
		rewritten = rewritten.Replace("declare i32 @printf(ptr, ...)\n\n", string.Empty,
			StringComparison.Ordinal);
		return EnsureWindowsPrintRuntimeSupport(rewritten);
	}

	private static string BuildNumericPrintReplacement(Match match,
		Dictionary<string, (int TextLength, int PrefixLength)> stringLengths, int replacementIndex)
	{
		var label = match.Groups["label"].Value;
		var value = match.Groups["value"].Value;
		var prefixLength = stringLengths[label].PrefixLength;
		return $"  %stdout_{replacementIndex} = call ptr @GetStdHandle(i32 -11)\n" +
			$"  %written_{replacementIndex} = alloca i32\n" +
			$"  call i32 @WriteFile(ptr %stdout_{replacementIndex}, ptr {label}, i32 {prefixLength}, ptr %written_{replacementIndex}, ptr null)\n" +
			$"  call void @print_number_from_double(ptr %stdout_{replacementIndex}, double {value})";
	}

	//ncrunch: no coverage start
	private static string BuildTextPrintReplacement(Match match,
		Dictionary<string, (int TextLength, int PrefixLength)> stringLengths, int replacementIndex)
	{
		var label = match.Groups["label"].Value;
		var textLength = stringLengths[label].TextLength;
		return $"  %stdout_{replacementIndex} = call ptr @GetStdHandle(i32 -11)\n" +
			$"  %written_{replacementIndex} = alloca i32\n" +
			$"  call i32 @WriteFile(ptr %stdout_{replacementIndex}, ptr {label}, i32 {textLength}, ptr %written_{replacementIndex}, ptr null)";
	} //ncrunch: no coverage end

	private static Dictionary<string, (int TextLength, int PrefixLength)> ParseStringLengths(
		string llvmIr)
	{
		var result = new Dictionary<string, (int TextLength, int PrefixLength)>(StringComparer.Ordinal);
		foreach (Match match in StringConstantRegex.Matches(llvmIr))
		{
			var encodedText = match.Groups["text"].Value;
			var nullIndex = encodedText.IndexOf("\\00", StringComparison.Ordinal);
			var placeholderIndex = encodedText.IndexOf("%g", StringComparison.Ordinal);
			var printableText = nullIndex > -1
				? encodedText[..nullIndex]
				: encodedText;
			var printablePrefix = placeholderIndex > -1
				? encodedText[..placeholderIndex]
				: printableText;
			result[match.Groups["label"].Value] =
				(CountEncodedBytes(printableText), CountEncodedBytes(printablePrefix));
		}
		return result;
	}

	private static int CountEncodedBytes(string encodedText)
	{
		var byteCount = 0;
		for (var index = 0; index < encodedText.Length; index++)
			if (encodedText[index] == '\\' && index + 2 < encodedText.Length &&
				byte.TryParse(encodedText.AsSpan(index + 1, 2), NumberStyles.HexNumber,
					CultureInfo.InvariantCulture, out _))
			{
				byteCount++;
				index += 2;
			}
			else
				byteCount++;
		return byteCount;
	}

	private static string EnsureWindowsPrintRuntimeSupport(string llvmIr)
	{
		var additions = new StringBuilder();
		if (!llvmIr.Contains("@_fltused = global i32 0", StringComparison.Ordinal))
			additions.AppendLine("@_fltused = global i32 0");
		if (!llvmIr.Contains("declare ptr @GetStdHandle(i32)", StringComparison.Ordinal))
			additions.AppendLine("declare ptr @GetStdHandle(i32)");
		if (!llvmIr.Contains("declare i32 @WriteFile(ptr, ptr, i32, ptr, ptr)",
			StringComparison.Ordinal))
			additions.AppendLine("declare i32 @WriteFile(ptr, ptr, i32, ptr, ptr)");
		if (additions.Length > 0)
		{
			var headerInsertIndex = llvmIr.IndexOf("\n\n", StringComparison.Ordinal);
			llvmIr = headerInsertIndex > -1
				? llvmIr.Insert(headerInsertIndex + 2, additions + "\n")
				: additions + "\n" + llvmIr;
		}
		if (!llvmIr.Contains("define void @print_number_from_double(", StringComparison.Ordinal))
		{
			var metadataIndex = llvmIr.IndexOf("\n!llvm.module.flags", StringComparison.Ordinal);
			var helper = "\n" + BuildWindowsPrintNumberHelper() + "\n";
			llvmIr = metadataIndex > -1
				? llvmIr.Insert(metadataIndex, helper)
				: llvmIr + helper;
		}
		return llvmIr;
	}

	private static string BuildWindowsPrintNumberHelper() =>
		string.Join("\n", "define void @print_number_from_double(ptr %stdout, double %value) {",
			"entry:",
			"  %buffer = alloca [64 x i8]",
			"  %bufferStart = getelementptr [64 x i8], ptr %buffer, i64 0, i64 0",
			"  %remainingPtr = alloca i64",
			"  %writeIndexPtr = alloca i64",
			"  %writtenPtr = alloca i32",
			"  %number = fptosi double %value to i64",
			"  %isNegative = icmp slt i64 %number, 0",
			"  %negated = sub i64 0, %number",
			"  %absolute = select i1 %isNegative, i64 %negated, i64 %number",
			"  store i64 %absolute, ptr %remainingPtr",
			"  store i64 62, ptr %writeIndexPtr",
			"  %newlinePtr = getelementptr i8, ptr %bufferStart, i64 62",
			"  store i8 10, ptr %newlinePtr",
			"  %isZero = icmp eq i64 %absolute, 0",
			"  br i1 %isZero, label %storeZero, label %digitLoop",
			"storeZero:",
			"  %zeroIndex = load i64, ptr %writeIndexPtr",
			"  %zeroStoreIndex = sub i64 %zeroIndex, 1",
			"  store i64 %zeroStoreIndex, ptr %writeIndexPtr",
			"  %zeroPtr = getelementptr i8, ptr %bufferStart, i64 %zeroStoreIndex",
			"  store i8 48, ptr %zeroPtr",
			"  br label %afterDigits",
			"digitLoop:",
			"  %current = load i64, ptr %remainingPtr",
			"  %remainder = urem i64 %current, 10",
			"  %quotient = udiv i64 %current, 10",
			"  store i64 %quotient, ptr %remainingPtr",
			"  %digitValue = add i64 %remainder, 48",
			"  %digitByte = trunc i64 %digitValue to i8",
			"  %loopIndex = load i64, ptr %writeIndexPtr",
			"  %digitStoreIndex = sub i64 %loopIndex, 1",
			"  store i64 %digitStoreIndex, ptr %writeIndexPtr",
			"  %digitPtr = getelementptr i8, ptr %bufferStart, i64 %digitStoreIndex",
			"  store i8 %digitByte, ptr %digitPtr",
			"  %hasMoreDigits = icmp ne i64 %quotient, 0",
			"  br i1 %hasMoreDigits, label %digitLoop, label %afterDigits",
			"afterDigits:",
			"  br i1 %isNegative, label %storeSign, label %prepareWrite",
			"storeSign:",
			"  %signIndex = load i64, ptr %writeIndexPtr",
			"  %signStoreIndex = sub i64 %signIndex, 1",
			"  store i64 %signStoreIndex, ptr %writeIndexPtr",
			"  %signPtr = getelementptr i8, ptr %bufferStart, i64 %signStoreIndex",
			"  store i8 45, ptr %signPtr",
			"  br label %prepareWrite",
			"prepareWrite:",
			"  %startIndex = load i64, ptr %writeIndexPtr",
			"  %outputPtr = getelementptr i8, ptr %bufferStart, i64 %startIndex",
			"  %length64 = sub i64 63, %startIndex",
			"  %length32 = trunc i64 %length64 to i32",
			"  call i32 @WriteFile(ptr %stdout, ptr %outputPtr, i32 %length32, ptr %writtenPtr, ptr null)",
			"  ret void",
			"}");

	private static readonly Regex StringConstantRegex =
		new(@"^(?<label>@[\w\.]+) = internal constant \[\d+ x i8\] c""(?<text>(?:[^""\\]|\\.)*)""",
			RegexOptions.Multiline);
	private static readonly Regex PrintWithNumberRegex =
		new(@"^\s*%[\w\.]+ = call i32 \(ptr, \.\.\.\) @printf\(ptr (?<label>@[\w\.]+), double (?<value>[^)]+)\)\s*$",
			RegexOptions.Multiline);
	private static readonly Regex PrintTextRegex =
		new(@"^\s*%[\w\.]+ = call i32 \(ptr, \.\.\.\) @printf\(ptr (?<label>@[\w\.]+)\)\s*$",
			RegexOptions.Multiline);
	public static bool IsAvailable =>
		ToolRunner.FindTool("mlir-opt") != null &&
		ToolRunner.FindTool("mlir-translate") != null &&
		ToolRunner.FindTool("clang") != null;
}