using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Running;

namespace Strict.Expressions.Tests;

[MemoryDiagnoser]
[SimpleJob(RunStrategy.Throughput, warmupCount: 1, iterationCount: 10)]
public class BooleanTests : TestExpressions
{
	[Test]
	public void ParseTrue() => ParseAndCheckOutputMatchesInput("true", new Boolean(method, true));

	[Test]
	public void ParseFalse() =>
		ParseAndCheckOutputMatchesInput("false", new Boolean(method, false));

	//ncrunch: no coverage start
	/// <summary>
	/// SpanIsTrueTextOrIsFalseText |   160.8 us |  1.64 us |  0.86 us |      1 KB |
	/// SpanEquals									|   283.7 us | 13.68 us |  9.05 us |      1 KB |
	/// SpanBooleanTryParse					| 1,730.8 us | 26.90 us | 17.79 us |      1 KB |
	/// </summary>
	[Test]
	[Category("Slow")]
	[Benchmark]
	// ReSharper disable MethodTooLong, just testing span here
	public void SpanIsTrueTextOrIsFalseText()
	{
		var case1 = "isOpen is true".AsSpan("isOpen is ".Length);
		var case2 = "isOpen is false".AsSpan("isOpen is ".Length);
		var case3 = "isOpen is false".AsSpan("isOpen ".Length);
		var case4 = "isOpen is false".AsSpan(0, "isOpen".Length);
		var case5 = "true".AsSpan();
		var counter = 0;
		for (var iteration = 0; iteration < 100000; iteration++)
		{
			if (case1.IsTrueText())
				counter--;
			else if (case1.IsFalseText())
				counter++;
			if (case2.IsTrueText())
				counter--;
			else if (case2.IsFalseText())
				counter++;
			if (case3.IsTrueText())
				counter--;
			else if (case3.IsFalseText())
				counter++;
			if (case4.IsTrueText())
				counter--;
			else if (case4.IsFalseText())
				counter++;
			if (case5.IsTrueText())
				counter--;
			else if (case5.IsFalseText())
				counter++;
		}
		//"abcef".AsSpan().IsFalseText();
		//"trued".AsSpan().IsFalseText();
		Assert.That(counter, Is.EqualTo(-100000));
	}

	[Test]
	[Category("Slow")]
	[Benchmark]
	public void SpanEquals()
	{
		var caseOne = "isOpen is true".AsSpan("isOpen is ".Length);
		var caseTwo = "isOpen is false".AsSpan("isOpen is ".Length);
		var caseThree = "isOpen is false".AsSpan("isOpen ".Length);
		var caseFour = "isOpen is false".AsSpan(0, "isOpen".Length);
		var caseFive = "true".AsSpan();
		var counter = 0;
		for (var iteration = 0; iteration < 100000; iteration++)
		{
			if (caseOne.Equals("true", StringComparison.Ordinal))
				counter--;
			else if (caseOne.Equals("false", StringComparison.Ordinal))
				counter++;
			if (caseTwo.Equals("true", StringComparison.Ordinal))
				counter--;
			else if (caseTwo.Equals("false", StringComparison.Ordinal))
				counter++;
			if (caseThree.Equals("true", StringComparison.Ordinal))
				counter--;
			else if (caseThree.Equals("false", StringComparison.Ordinal))
				counter++;
			if (caseFour.Equals("true", StringComparison.Ordinal))
				counter--;
			else if (caseFour.Equals("false", StringComparison.Ordinal))
				counter++;
			if (caseFive.Equals("true", StringComparison.Ordinal))
				counter--;
			else if (caseFive.Equals("false", StringComparison.Ordinal))
				counter++;
		}
		Assert.That(counter, Is.EqualTo(-100000));
	}

	[Test]
	[Category("Slow")]
	[Benchmark]
	public void SpanBooleanTryParse()
	{
		var firstCase = "isOpen is true".AsSpan("isOpen is ".Length);
		var secondCase = "isOpen is false".AsSpan("isOpen is ".Length);
		var thirdCase = "isOpen is false".AsSpan("isOpen ".Length);
		var fourthCase = "isOpen is false".AsSpan(0, "isOpen".Length);
		var fifthCase = "true".AsSpan();
		var firstCounter = 0;
		var secondCounter = 0;
		var thirdCounter = 0;
		var fourthCounter = 0;
		var fifthCounter = 0;
		const int NumberOfIterations = 100000;
		for (var iteration = 0; iteration < NumberOfIterations; iteration++)
		{
			if (bool.TryParse(firstCase, out var result1))
				firstCounter += result1
					? 1
					: -1;
			if (bool.TryParse(secondCase, out var result2))
				secondCounter += result2
					? 1
					: -1;
			if (bool.TryParse(thirdCase, out var result3))
				thirdCounter += result3
					? 1
					: -1;
			if (bool.TryParse(fourthCase, out var result4))
				fourthCounter += result4
					? 1
					: -1;
			if (bool.TryParse(fifthCase, out var result5))
				fifthCounter += result5
					? 1
					: -1;
		}
		Assert.That(firstCounter, Is.EqualTo(NumberOfIterations));
		Assert.That(secondCounter, Is.EqualTo(-NumberOfIterations));
		Assert.That(thirdCounter, Is.EqualTo(0));
		Assert.That(fourthCounter, Is.EqualTo(0));
		Assert.That(fifthCounter, Is.EqualTo(NumberOfIterations));
	}

	[Test]
	[Category("Manual")]
	public void BenchmarkCompare() => BenchmarkRunner.Run<BooleanTests>();
}