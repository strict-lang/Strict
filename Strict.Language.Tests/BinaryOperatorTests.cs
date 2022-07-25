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
	private const string Input = @"has numbers
GetComplicatedSequenceTexts returns Texts
	ConvertingNumbers(1, 21).GetComplicatedSequenceTexts is (""7"", ""16"")
	return for numbers
		 to Text
		Length * Length
		4 + value * 3
		 to Text";
	private readonly ReadOnlyMemory<char> inputMemory = Input.AsMemory();

	[Benchmark]
	[Test]
	public void IsOperatorUsingStringContains()
	{
		var operatorCounter = 0;
		for (var iteration = 0; iteration < 1000; iteration++)
		{
			operatorCounter = 0;
			foreach (var word in Input.SplitWords())
				if (word.IsOperator())
					operatorCounter++;
		}
		Assert.That(operatorCounter, Is.EqualTo(6));
	}

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
		Assert.That(operatorCounter, Is.EqualTo(6));
	}

	[Category("Manual")]
	[Test]
	public void BenchmarkIsOperator() => BenchmarkRunner.Run<BinaryOperatorTests>();
}