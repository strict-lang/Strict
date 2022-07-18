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
			foreach (var line in Input.SplitLines())
			foreach (var word in line.SplitWords())
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
			//foreach (var line in inputMemory.Span.EnumerateLines())
			{
				//TODO: put in SpanExtensions
				var line = inputMemory.Span;
				var spaceIndex = line.IndexOf(' ');
				//Console.WriteLine("line="+line.ToString()+", spaceIndex="+spaceIndex);
				if (spaceIndex < 0 || spaceIndex + 1 >= line.Length)
					continue;
				var offset = 0;
				//ignore first word
				do
				{
					offset += spaceIndex + 1;
					var restSlice = line.Slice(offset);
					spaceIndex = restSlice.IndexOf(' ');
					if (spaceIndex < 0)
						spaceIndex = line.Length - offset;
					//Console.WriteLine("offset=" + offset + ", spaceIndex=" + spaceIndex);
					var word = line.Slice(offset, spaceIndex);
					//Console.WriteLine("second word: " + word.ToString());
					if (word.IsOperator())
						operatorCounter++;
				} while (offset + spaceIndex < line.Length - 1);
				/*slow, only 40% faster ..
				foreach (var word in line.Split())
					if (line.Slice(word.Start.Value, word.End.Value - word.Start.Value).IsOperator())
						operatorCounter++;
				*/
			}
		}
		Assert.That(operatorCounter, Is.EqualTo(6));
	}

	[Category("Manual")]
	[Test]
	public void BenchmarkIsOperator() => BenchmarkRunner.Run<BinaryOperatorTests>();
}