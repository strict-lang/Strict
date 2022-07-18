using System.ComponentModel;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Strict.Language;

namespace Benchmark;

internal class Program
{
	static void Main()
	{
	}
}

////[MemoryDiagnoser]
////public class BinaryOperatorTests
////{
////	[Benchmark]
////	public void IsOperatorUsingStringContains()
////	{
////		for (var index = 0; index < 1000000; index++)
////		{
////			"+".IsOperator();
////			"*".IsOperator();
////			"/".IsOperator();
////			"-".IsOperator();
////			" randomwasdasdasdadasd +fewgfewrgfergergsdasda/dsa * ".IsOperator();
////		}
////	}

////	[Benchmark]
////	public void IsOperatorUsingFirstOrDefault()
////	{
////		for (var index = 0; index < 1000000; index++)
////		{
////			"+".IsOperator();
////			"*".IsOperator();
////			"/".IsOperator();
////			"-".IsOperator();
////			" randomwasdasdasdadasd +fewgfewrgfergergsdasda/dsa * ".FindFirstOperator();
////		}
////	}

////	[Benchmark]
////	public void IsOperatorUsingReadOnlyMemory()
////	{
////		for (var index = 0; index < 1000000; index++)
////		{
////			"+".AsMemory().IsOperator();
////			"*".AsMemory().IsOperator();
////			"/".AsMemory().IsOperator();
////			"-".AsMemory().IsOperator();
////			" randomwasdasdasdadasd +fewgfewrgfergergsdasda/dsa * ".AsMemory().IsOperator();
////		}
////	}
//}