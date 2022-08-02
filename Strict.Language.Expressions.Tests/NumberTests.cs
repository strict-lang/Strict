using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Running;
using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

[MemoryDiagnoser]
[SimpleJob(RunStrategy.Throughput, warmupCount: 1, targetCount: 10)]
public class NumberTests : TestExpressions
{
	[Test]
	public void NumberWithCharacters() => Assert.That(() => ParseExpression("7abc"), Throws.InstanceOf<UnknownExpression>());

	[Test]
	public void ParseNumber() => ParseAndCheckOutputMatchesInput("77", new Number(method, 77));

	[Test]
	public void TwoNumbersWithTheSameValueAreTheSame() =>
		Assert.That(new Number(method, 5), Is.EqualTo(new Number(method, 5)));

	[TestCase("7.05")]
	[TestCase("0.5")]
	[TestCase("77")]
	public void ValidNumbers(string input) => Assert.That(ParseExpression(input), Is.EqualTo(new Number(method, double.Parse(input))));

	/// <summary>
	///		Method			 |     Mean |    Error |   StdDev | Allocated
	/// IntTryParse		 | 11.30 ms | 0.123 ms | 0.064 ms |      1 KB |
	/// DoubleTryParse | 46.19 ms | 0.371 ms | 0.221 ms |      1 KB |
	/// </summary>
	[Category("Manual")]
	[Benchmark]
	[Test]
	public void IntTryParse()
	{
		const string Case1 = "7";
		const string Case2 = "7.59";
		const string Case3 = "text";
		const string Case4 = "0.5";
		const string Case5 = "-50.5";
		const string Case6 = "5045142575";
		const string Case7 = "2000000102";
		var case8 = "50".AsSpan();
		var case9 = "2000000102".AsSpan();
		var counter = 0;
		for (var iteration = 0; iteration < 100000; iteration++)
		{
			if (int.TryParse(Case1, out var result1))
				if (result1 == 7)
					counter++;
			if (int.TryParse(Case2, out _))
				counter++; //ncrunch: no coverage start
			if (int.TryParse(Case3, out _))
				counter++;
			if (int.TryParse(Case4, out _))
				counter++;
			if (int.TryParse(Case5, out _))
				counter++;
			if (int.TryParse(Case6, out _))
				counter++; //ncrunch: no coverage end
			if (int.TryParse(Case7, out var result7))
				if (result7 == 2000000102)
					counter++;
			if (int.TryParse(case8, out var result8))
				if (result8 == 50)
					counter++;
			if (int.TryParse(case9, out var result9))
				if (result9 == 2000000102)
					counter++;
		}
		Assert.That(counter, Is.EqualTo(400000));
	}

	[Category("Manual")]
	[Benchmark]
	[Test]
	public void DoubleTryParse()
	{
		const string Case1 = "7";
		const string Case2 = "7.59";
		const string Case3 = "text";
		const string Case4 = "0.5";
		const string Case5 = "-50.5";
		const string Case6 = "5045142575";
		const string Case7 = "2000000102";
		var case8 = "50".AsSpan();
		var case9 = "2000000102".AsSpan();
		var counter = 0;
		for (var iteration = 0; iteration < 100000; iteration++)
		{
			if (double.TryParse(Case1, out var result1))
				if (result1 == 7)
					counter++;
			if (double.TryParse(Case2, out _))
				counter++;
			if (double.TryParse(Case3, out _))
				counter++; //ncrunch: no coverage
			if (double.TryParse(Case4, out _))
				counter++;
			if (double.TryParse(Case5, out _))
				counter++;
			if (double.TryParse(Case6, out _))
				counter++;
			if (double.TryParse(Case7, out var result7))
				if (result7 == 2000000102)
					counter++;
			if (double.TryParse(case8, out var result8))
				if (result8 == 50)
					counter++;
			if (double.TryParse(case9, out var result9))
				if (result9 == 2000000102)
					counter++;
		}
		Assert.That(counter, Is.EqualTo(800000));
	}

	[Category("Manual")]
	[Test]
	public void BenchmarkCompare() => BenchmarkRunner.Run<NumberTests>();
}