using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Running;
using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

[MemoryDiagnoser]
[SimpleJob(RunStrategy.Throughput, warmupCount: 1, targetCount: 10)]
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
	///                  SpanEquals |   283.7 us | 13.68 us |  9.05 us |      1 KB |
	///         SpanBooleanTryParse | 1,730.8 us | 26.90 us | 17.79 us |      1 KB |
	/// </summary>
	[Benchmark]
	[Test]
	// ReSharper disable MethodTooLong
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

	[Benchmark]
	[Test]
	public void SpanEquals()
	{
		var case1 = "isOpen is true".AsSpan("isOpen is ".Length);
		var case2 = "isOpen is false".AsSpan("isOpen is ".Length);
		var case3 = "isOpen is false".AsSpan("isOpen ".Length);
		var case4 = "isOpen is false".AsSpan(0, "isOpen".Length);
		var case5 = "true".AsSpan();
		var counter = 0;
		for (var iteration = 0; iteration < 100000; iteration++)
		{
			if (case1.Equals("true", StringComparison.Ordinal))
				counter--;
			else if (case1.Equals("false", StringComparison.Ordinal))
				counter++;
			if (case2.Equals("true", StringComparison.Ordinal))
				counter--;
			else if (case2.Equals("false", StringComparison.Ordinal))
				counter++;
			if (case3.Equals("true", StringComparison.Ordinal))
				counter--;
			else if (case3.Equals("false", StringComparison.Ordinal))
				counter++;
			if (case4.Equals("true", StringComparison.Ordinal))
				counter--;
			else if (case4.Equals("false", StringComparison.Ordinal))
				counter++;
			if (case5.Equals("true", StringComparison.Ordinal))
				counter--;
			else if (case5.Equals("false", StringComparison.Ordinal))
				counter++;
		}
		Assert.That(counter, Is.EqualTo(-100000));
	}

	[Benchmark]
	[Test]
	public void SpanBooleanTryParse()
	{
		var case1 = "isOpen is true".AsSpan("isOpen is ".Length);
		var case2 = "isOpen is false".AsSpan("isOpen is ".Length);
		var case3 = "isOpen is false".AsSpan("isOpen ".Length);
		var case4 = "isOpen is false".AsSpan(0, "isOpen".Length);
		var case5 = "true".AsSpan();
		var counter = 0;
		for (var iteration = 0; iteration < 100000; iteration++)
		{
			if (bool.TryParse(case1, out var result1))
			{
				if (result1)
					counter--;
				else
					counter++;
			}
			if (bool.TryParse(case2, out var result2))
			{
				if (result2)
					counter--;
				else
					counter++;
			}
			if (bool.TryParse(case3, out var result3))
			{
				if (result3)
					counter--;
				else
					counter++;
			}
			if (bool.TryParse(case4, out var result4))
			{
				if (result4)
					counter--;
				else
					counter++;
			}
			if (bool.TryParse(case5, out var result5))
			{
				if (result5)
					counter--;
				else
					counter++;
			}
		}
		Assert.That(counter, Is.EqualTo(-100000));
	}

	[Category("Manual")]
	[Test]
	public void BenchmarkCompare() => BenchmarkRunner.Run<BooleanTests>();
}