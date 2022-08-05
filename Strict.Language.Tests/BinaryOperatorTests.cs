using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Running;
using NUnit.Framework;

namespace Strict.Language.Tests;

[MemoryDiagnoser]
[SimpleJob(RunStrategy.Throughput, warmupCount: 1, targetCount: 10)]
public class BinaryOperatorTests
{
	/// <summary>
	/// IsOperatorUsingStringContains | 1,325.0 us | 34.42 us | 20.49 us | 251.9531 |  1,032 KB |
	/// IsOperatorUsingReadOnlyMemory |   420.7 us | 13.87 us |  8.26 us |        - |      1 KB |
	/// </summary>
	/// Removed string contains operator method as it is not being used anymore in strict
	private const string Input = @"has numbers
GetComplicatedSequenceTexts returns Texts
	ConvertingNumbers(1, 21).GetComplicatedSequenceTexts is (""7"", ""16"")
	return for numbers
		 to Text
		Length * Length
		4 + value * 3 ^ 2
		 to Text";
	private readonly ReadOnlyMemory<char> inputMemory = Input.AsMemory();

	[Benchmark]
	[Test]
	public void IsOperatorUsingReadOnlyMemory()
	{
		var operatorCounter = 0;
		for (var iteration = 0; iteration < 1000; iteration++)
		{
			operatorCounter = 0;
			foreach (var word in inputMemory.Span.Split())
				if (word[0].IsSingleCharacterOperator() || word.IsMultiCharacterOperator())
					operatorCounter++;
		}
		Assert.That(operatorCounter, Is.EqualTo(7));
	}

	//ncrunch: no coverage start
	[Category("Manual")]
	[Test]
	public void BenchmarkIsOperator() => BenchmarkRunner.Run<BinaryOperatorTests>();
}