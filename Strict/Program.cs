using System.Runtime;
using Strict.Bytecode;
using Strict.Compiler;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict;

public static class Program
{
	//ncrunch: no coverage start
	public static async Task<int> Main(string[] args)
	{
		ProfileOptimization.SetProfileRoot(AppContext.BaseDirectory);
		ProfileOptimization.StartProfile(nameof(Strict) + ".jitprofile");
		args = ResolveImplicitExecutableTarget(args);
		var command = args.Length > 0 && Commands.Contains(args[0])
			? args[0]
			: "";
		var arguments = command == ""
			? args
			: args[1..];
		if (arguments.Length == 0)
		{
			DisplayUsageInformation();
			return args.Length == 0
				? 0
				: UsageError;
		}
		var options = arguments.Skip(1).Where(IsOption).ToHashSet(StringComparer.OrdinalIgnoreCase);
		var unknownOption = options.FirstOrDefault(option => !KnownOptions.Contains(option));
		if (unknownOption != null)
		{
			Console.WriteLine("Unknown option " + unknownOption +
				", run Strict without arguments to see all commands and options");
			return UsageError;
		}
		try
		{
			await Run(command, arguments[0], options,
				arguments.Skip(1).Where(arg => !IsOption(arg)).ToArray());
			return 0;
		}
		catch (Exception ex)
		{
			Console.WriteLine(DescribeFailure(ex, options.Contains("-diagnostics")));
			return ex is Runner.StrictFileNotFound
				? UsageError
				: 1;
		}
	}

	private static readonly string[] Commands = ["run", "check", "test", "build", "decompile"];
	private static readonly HashSet<string> KnownOptions = new([
		"-Windows", "-Linux", "-MacOS", "-mlir", "-llvm", "-nasm", "-diagnostics", "-profile",
		"-decompile"
	], StringComparer.OrdinalIgnoreCase);
	private const int UsageError = 2;

	private static bool IsOption(string argument) =>
		argument.Length > 1 && argument[0] == '-' && char.IsLetter(argument[1]);

	/// <summary>
	/// Strict errors already point to the .strict source lines, .NET frames only help internal bugs.
	/// </summary>
	private static string DescribeFailure(Exception ex, bool diagnostics) =>
		diagnostics || !IsStrictError(ex)
			? "Execution failed: " + ex
			: ex.GetType().Name + ": " + ex.Message + (ex.InnerException == null
				? ""
				: Environment.NewLine + DescribeFailure(ex.InnerException, false));

	private static bool IsStrictError(Exception? ex) =>
		ex == null || ex.GetType().Namespace?.StartsWith(nameof(Strict), StringComparison.Ordinal) ==
		true && IsStrictError(ex.InnerException);

	private static void DisplayUsageInformation() =>
		Console.WriteLine("""
											Usage: Strict [command] <file.strict|.strictbinary|folder> [-options] [args...]

											Commands (default if nothing specified: run)
											  run          Build the .strictbinary cache if needed and execute Run in the VM
											  check        Parse and validate the type, no tests and no execution
											  test         check and run all inline method tests
											  build        Compile to a native executable for this platform (or -Windows/-Linux/-MacOS)
											  decompile    Decompile a .strictbinary into partial .strict source files
											Exit codes: 0 success, 1 failed (parsing, test or runtime error), 2 wrong usage

											Options
											  -Windows     Compile to a native Windows x64 optimized executable (.exe)
											  -Linux       Compile to a native Linux x64 optimized executable
											  -MacOS       Compile to a native macOS x64 optimized executable
											  -mlir        Force MLIR backend (default, requires mlir-opt + mlir-translate + clang)
											               MLIR is the default, best optimized, uses parallel CPU and GPU (Cuda) execution
											  -llvm        Force LLVM IR backend (fallback, requires clang: https://releases.llvm.org)
											  -nasm        Force NASM backend (fallback, less optimized, requires nasm + gcc/clang)
											  -diagnostics Output detailed step-by-step logs and timing for each pipeline stage
											               (automatically enabled in Debug builds)
											  -profile     After running print the slowest invoked methods (time, calls)
											  -decompile   Decompile a .strictbinary into partial .strict source files
											               (creates a folder with one .strict per type; no tests, optimized)

											Arguments:
											  args...      Optional text or numbers passed to called method
											               Example to call Run method: Strict Sum.strict 5 10 20 => prints 35
											               Example to call any expression, must contain brackets: (1, 2, 3).Length => 3

											Examples:
											  Strict Examples/SimpleCalculator.strict
											  Strict test Examples/SimpleCalculator.strict
											  Strict build Examples/SimpleCalculator.strict
											  Strict Examples/SimpleCalculator.strict -Windows
											  Strict Examples/SimpleCalculator.strict -diagnostics
											  Strict Examples/SimpleCalculator.strictbinary
											  Strict Examples/SimpleCalculator.strictbinary -decompile
											  Strict Examples/Sum.strict 5 10 20
											  Strict List.strict (1, 2, 3).Length

											Notes:
												Only .strict files contain the full actual code, everything after that is stripped,
												optimized, and just includes what is actually executed (.strictbinary is much smaller).
											  Always caches bytecode into a .strictbinary for fast subsequent execution.
											  .strictbinary files are reused when they are newer than all of the used source files.
											""");

	private static async Task Run(string command, string filePath, ICollection<string> options,
		string[] programArguments)
	{
		if (command == "decompile" || options.Contains("-decompile"))
		{
			var outputFolder = Path.GetFileNameWithoutExtension(filePath);
			new Decompiler().Decompile(new BinaryExecutable(filePath), outputFolder);
			Console.WriteLine("Decompilation complete, written partial .strict files (no tests, only " +
				"bytecode reconstruction) to folder:" + Environment.NewLine + outputFolder);
			return;
		}
		var diagnostics = IsDebugBuild || options.Contains("-diagnostics");
		var runner = new Runner(filePath, programArguments.Length == 0
			? Method.Run
			: string.Join(" ", programArguments), diagnostics)
		{
			Profile = options.Contains("-profile")
		};
		var platform = GetPlatformOption(options);
		if (command is "check" or "test")
			await runner.Check(command == "test");
		else if (command == "build" || platform.HasValue)
			await runner.Build(platform ?? CurrentPlatform, options.Contains("-nasm")
				? CompilerBackend.Nasm
				: options.Contains("-llvm")
					? CompilerBackend.Llvm
					: CompilerBackend.MlirDefault);
		else
			await runner.Run();
	}

	private static bool IsDebugBuild =>
#if DEBUG
		true;
#else
		false;
#endif
	private static Platform CurrentPlatform =>
		OperatingSystem.IsWindows()
			? Platform.Windows
			: OperatingSystem.IsMacOS()
				? Platform.MacOS
				: Platform.Linux;

	private static string[] ResolveImplicitExecutableTarget(string[] args)
	{
		if (args.Length > 0 && (args[0].EndsWith(Type.Extension, StringComparison.OrdinalIgnoreCase) ||
			args[0].EndsWith(BinaryExecutable.Extension, StringComparison.OrdinalIgnoreCase) ||
			Directory.Exists(args[0])))
			return args;
		var processPath = Environment.ProcessPath;
		if (string.IsNullOrEmpty(processPath))
			return args;
		var implicitBinaryPath = Path.ChangeExtension(processPath, BinaryExecutable.Extension);
		if (File.Exists(implicitBinaryPath))
			return [implicitBinaryPath, .. args];
		var implicitSourcePath = Path.ChangeExtension(processPath, Type.Extension);
		return File.Exists(implicitSourcePath)
			? [implicitSourcePath, .. args]
			: args;
	}

	private static Platform? GetPlatformOption(ICollection<string> options)
	{
		if (options.Contains("-Windows"))
			return Platform.Windows;
		if (options.Contains("-Linux"))
			return Platform.Linux;
		if (options.Contains("-MacOS"))
			return Platform.MacOS;
		return null;
	}
}