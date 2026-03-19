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
    var binaryFilePath = await GetExamplesBinaryFileAsync("SimpleCalculator");
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
      File.Delete(asmPath); //ncrunch: no coverage
    var binaryPath = await GetExamplesBinaryFileAsync("SimpleCalculator");
    await new Runner(binaryPath, TestPackage.Instance).Run();
    Assert.That(File.Exists(asmPath), Is.False);
  }

  [Test]
  public async Task SaveStrictBinaryWithTypeBytecodeEntriesOnlyAsync()
  {
    var binaryPath = await GetExamplesBinaryFileAsync("SimpleCalculator");
    using var archive = ZipFile.OpenRead(binaryPath);
    var entries = archive.Entries.Select(entry => entry.FullName.Replace('\\', '/')).ToList();
    Assert.That(entries.All(entry => entry.EndsWith(BytecodeSerializer.BytecodeEntryExtension,
      StringComparison.OrdinalIgnoreCase)), Is.True);
    Assert.That(entries.Any(entry => entry.Contains("#", StringComparison.Ordinal)), Is.False);
    Assert.That(entries, Does.Contain("SimpleCalculator.bytecode"));
    Assert.That(entries, Does.Contain("Strict/Number.bytecode"));
    Assert.That(entries, Does.Contain("Strict/Logger.bytecode"));
    Assert.That(entries, Does.Contain("Strict/Text.bytecode"));
    Assert.That(entries, Does.Contain("Strict/Character.bytecode"));
    Assert.That(entries, Does.Contain("Strict/TextWriter.bytecode"));
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

  [Test]
  public async Task RunSimpleCalculatorTwiceWithoutTestPackage()
  {
    await new Runner(SimpleCalculatorFilePath).Run();
    writer.GetStringBuilder().Clear();
    await new Runner(SimpleCalculatorFilePath).Run();
    Assert.That(writer.ToString(), Does.Contain("2 + 3 = 5"));
  }

  [Test]
  public async Task SaveStrictBinaryEntryNameTableSkipsPrefilledNames()
  {
    var tempDirectory = Path.Combine(Path.GetTempPath(), "Strict" + Guid.NewGuid().ToString("N"));
    Directory.CreateDirectory(tempDirectory);
    try
    {
      var sourceCopyPath = Path.Combine(tempDirectory, Path.GetFileName(SimpleCalculatorFilePath));
      File.Copy(SimpleCalculatorFilePath, sourceCopyPath);
      await new Runner(sourceCopyPath, TestPackage.Instance).Run();
      var binaryPath = Path.ChangeExtension(sourceCopyPath, BytecodeSerializer.Extension);
      using var archive = ZipFile.OpenRead(binaryPath);
      var entry = archive.Entries.First(file => file.FullName == "SimpleCalculator.bytecode");
      using var reader = new BinaryReader(entry.Open());
      Assert.That(reader.ReadByte(), Is.EqualTo((byte)'S'));
      Assert.That(reader.ReadByte(), Is.EqualTo(BytecodeSerializer.Version));
      var customNamesCount = reader.Read7BitEncodedInt();
      var customNames = new List<string>(customNamesCount);
      for (var nameIndex = 0; nameIndex < customNamesCount; nameIndex++)
        customNames.Add(reader.ReadString());
      Assert.That(customNames, Does.Not.Contain("Strict/Number"));
      Assert.That(customNames, Does.Not.Contain("Strict/Text"));
      Assert.That(customNames, Does.Not.Contain("Strict/Boolean"));
      Assert.That(customNames, Does.Not.Contain("SimpleCalculator"));
    }
    finally
    {
      if (Directory.Exists(tempDirectory))
        Directory.Delete(tempDirectory, true);
    }
  }

  private static string SimpleCalculatorFilePath => GetExamplesFilePath("SimpleCalculator");
  private static string SumFilePath => GetExamplesFilePath("Sum");

  private async Task<string> GetExamplesBinaryFileAsync(string filename)
  {
    var localPath = Path.ChangeExtension(GetExamplesFilePath(filename), BytecodeSerializer.Extension);
    if (!File.Exists(localPath))
      await new Runner(GetExamplesFilePath(filename), TestPackage.Instance).Run(); //ncrunch: no coverage
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
