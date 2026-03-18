using Strict.Bytecode.Serialization;
using Strict.Compiler;
using Strict.Compiler.Assembly;
using Strict.Language;
using Strict.Language.Tests;
using System.IO.Compression;

namespace Strict.Tests;

public sealed class RunnerTests
{
  [SetUp]
  public void CreateTextWriter()
  {
    writer = new StringWriter();
    rememberConsole = Console.Out;
    Console.SetOut(writer);
  }

  private StringWriter writer = null!;
  private TextWriter rememberConsole = null!;

  [TearDown]
  public void RestoreConsole()
  {
    Console.SetOut(rememberConsole);
    CleanupGeneratedFiles();
  }

  private static void CleanupGeneratedFiles()
  {
    var examplesDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..",
      "Examples");
    if (!Directory.Exists(examplesDir))
      return;
    foreach (var extension in new[] { ".ll", ".mlir", ".llvm.mlir", ".asm", ".obj", ".exe", ".strictbinary" })
      foreach (var file in Directory.GetFiles(examplesDir, "*" + extension))
        File.Delete(file);
  }

  [Test]
  public async Task RunSimpleCalculator()
  {
    var asmFilePath = Path.ChangeExtension(SimpleCalculatorFilePath, ".asm");
    if (File.Exists(asmFilePath))
      File.Delete(asmFilePath);
    await new Runner(SimpleCalculatorFilePath, TestPackage.Instance).Run();
    Assert.That(writer.ToString(),
      Does.StartWith("2 + 3 = 5" + Environment.NewLine + "2 * 3 = 6" + Environment.NewLine));
    Assert.That(File.Exists(asmFilePath), Is.False);
  }

  [Test]
  public async Task RunFromBytecodeFileProducesSameOutput()
  {
    var binaryFilePath = GetExamplesBinaryFile("SimpleCalculator");
    await new Runner(binaryFilePath, TestPackage.Instance).Run();
    Assert.That(writer.ToString(),
      Does.StartWith("2 + 3 = 5" + Environment.NewLine + "2 * 3 = 6"));
  }

  [Test]
  public async Task RunFromBytecodeFileWithoutStrictSourceFile()
  {
    var tempDirectory = Path.Combine(Path.GetTempPath(), "Strict" + Guid.NewGuid().ToString("N"));
    Directory.CreateDirectory(tempDirectory);
    var copiedSourceFilePath = Path.Combine(tempDirectory, Path.GetFileName(SimpleCalculatorFilePath));
    var copiedBinaryFilePath = Path.ChangeExtension(copiedSourceFilePath, BytecodeSerializer.Extension);
    try
    {
      File.Copy(SimpleCalculatorFilePath, copiedSourceFilePath);
      await new Runner(copiedSourceFilePath, TestPackage.Instance).Run();
      Assert.That(File.Exists(copiedBinaryFilePath), Is.True);
      writer.GetStringBuilder().Clear();
      File.Delete(copiedSourceFilePath);
      await new Runner(copiedBinaryFilePath, TestPackage.Instance).Run();
      Assert.That(writer.ToString(),
        Does.StartWith("2 + 3 = 5" + Environment.NewLine + "2 * 3 = 6"));
    }
    finally
    {
      if (Directory.Exists(tempDirectory))
        Directory.Delete(tempDirectory, true);
    }
  }

  [Test]
  public void BuildWithExpressionEntryPointThrows()
  {
    var runner = new Runner(SimpleCalculatorFilePath, TestPackage.Instance, "(1, 2, 3).Length");
    Assert.That(async () => await runner.Build(Platform.Windows),
      Throws.TypeOf<Runner.CannotBuildExecutableWithCustomExpression>());
  }

  [Test]
  public async Task AsmFileIsNotCreatedWhenRunningFromPrecompiledBytecode()
  {
    var asmPath = Path.ChangeExtension(SimpleCalculatorFilePath, ".asm");
    if (File.Exists(asmPath))
      File.Delete(asmPath);
    var binaryPath = GetExamplesBinaryFile("SimpleCalculator");
    await new Runner(binaryPath, TestPackage.Instance).Run();
    Assert.That(File.Exists(asmPath), Is.False);
  }

  [Test]
  public void SaveStrictBinaryWithTypeBytecodeEntriesOnly()
  {
    var binaryPath = GetExamplesBinaryFile("SimpleCalculator");
    using var archive = ZipFile.OpenRead(binaryPath);
    var entries = archive.Entries.Select(entry => entry.FullName).ToList();
    Assert.That(entries.All(entry => entry.EndsWith(BytecodeSerializer.BytecodeEntryExtension,
      StringComparison.OrdinalIgnoreCase)), Is.True);
    Assert.That(entries.Any(entry => entry.Contains("#", StringComparison.Ordinal)), Is.False);
    Assert.That(entries.Any(entry => entry.EndsWith("SimpleCalculator.bytecode",
      StringComparison.Ordinal)), Is.True);
  }

  [Test]
  public async Task RunSumWithProgramArguments()
  {
    await new Runner(SumFilePath, TestPackage.Instance, "5 10 20").Run();
    Assert.That(writer.ToString(), Does.Contain("35"));
  }

  [Test]
  public async Task RunSumWithDifferentProgramArgumentsDoesNotReuseCachedEntryPoint()
  {
    await new Runner(SumFilePath, TestPackage.Instance, "5 10 20").Run();
    writer.GetStringBuilder().Clear();
    await new Runner(SumFilePath, TestPackage.Instance, "1 2").Run();
    Assert.That(writer.ToString(), Does.Contain("3"));
  }

  [Test]
  public async Task RunSumWithNoArgumentsUsesEmptyList()
  {
    await new Runner(SumFilePath, TestPackage.Instance, "0").Run();
    Assert.That(writer.ToString(), Does.Contain("0"));
  }

  [Test]
  public async Task RunFibonacciRunner()
  {
    await new Runner(GetExamplesFilePath("FibonacciRunner"), TestPackage.Instance).Run();
    var output = writer.ToString();
    Assert.That(output, Does.Contain("Fibonacci(10) = 55"));
    Assert.That(output, Does.Contain("Fibonacci(5) = 2"));
  }

  private static string SimpleCalculatorFilePath => GetExamplesFilePath("SimpleCalculator");
  private static string SumFilePath => GetExamplesFilePath("Sum");

  private string GetExamplesBinaryFile(string filename)
  {
    var localPath = Path.ChangeExtension(GetExamplesFilePath(filename), BytecodeSerializer.Extension);
    if (File.Exists(localPath))
      File.Delete(localPath);
    new Runner(GetExamplesFilePath(filename), TestPackage.Instance).Run().GetAwaiter().GetResult();
    writer.GetStringBuilder().Clear();
    return localPath;
  }

  public static string GetExamplesFilePath(string filename)
  {
    var localPath = Path.Combine(
      Repositories.GetLocalDevelopmentPath(Repositories.StrictOrg, nameof(Strict)),
      "Examples", filename + Language.Type.Extension);
    return File.Exists(localPath)
      ? localPath
      : Path.Combine(FindRepoRoot(), "Examples", filename + Language.Type.Extension);
  }

  private static string FindRepoRoot()
  {
    var directory = AppContext.BaseDirectory;
    while (directory != null)
    {
      if (File.Exists(Path.Combine(directory, "Strict.sln")))
        return directory;
      directory = Path.GetDirectoryName(directory);
    }
    throw new DirectoryNotFoundException("Cannot find repository root (Strict.sln not found)");
  }
}
