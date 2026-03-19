using BenchmarkDotNet.Attributes;
using Strict.Bytecode;
using Strict.Bytecode.Tests;
using Strict.Expressions;
using Strict.Language.Tests;
using Microsoft.VSDiagnostics;

namespace Strict.Tests;
[CPUUsageDiagnoser]
public class VirtualMachineMutableListBenchmarks : TestBytecode
{
    private BinaryExecutable executable = null!;
    [GlobalSetup]
    public void Setup()
    {
        var source = new[]
        {
            "has count Number",
            "AddMany Numbers",
            "\tmutable myList = (0)",
            "\tfor count",
            "\t\tmyList = myList + value",
            "\tmyList"
        };
        executable = new BinaryGenerator(GenerateMethodCallFromSource(nameof(VirtualMachineMutableListBenchmarks), $"{nameof(VirtualMachineMutableListBenchmarks)}(100).AddMany", source)).Generate();
    }

    [Benchmark]
    public ValueInstance AddHundredElementsToMutableList() => new VirtualMachine(executable).Execute(executable.EntryPoint, initialVariables: null).Returns!.Value;
}