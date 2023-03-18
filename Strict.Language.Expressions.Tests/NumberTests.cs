using System;
using System.Globalization;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Running;
using NUnit.Framework;

namespace Strict.Language.Expressions.Tests;

[MemoryDiagnoser]
[SimpleJob(RunStrategy.Throughput, warmupCount: 1, iterationCount: 10)]
public class NumberTests : TestExpressions
{
	[Test]
	public void NumberWithCharacters() =>
		Assert.That(() => ParseExpression("7abc"), Throws.InstanceOf<UnknownExpression>());

	[Test]
	public void ParseNumber() => ParseAndCheckOutputMatchesInput("77", new Number(method, 77));

	[Test]
	public void TwoNumbersWithTheSameValueAreTheSame() =>
		Assert.That(new Number(method, 5), Is.EqualTo(new Number(method, 5)));

	[TestCase(Case1)]
	[TestCase(Case2)]
	[TestCase(Case3)]
	[TestCase(Case4)]
	[TestCase(Case5)]
	[TestCase(Case6)]
	[TestCase(Case7)]
	[TestCase(Case8)]
	[TestCase(Case9)]
	[TestCase("2.3e10")]
	[TestCase("1e-10")]
	[TestCase("7.5e+13")]
	public void ValidNumbers(string input) =>
		Assert.That(ParseExpression(input), Is.EqualTo(new Number(method, double.Parse(input, CultureInfo.InvariantCulture))));

	[Test]
	public void ParseTextToNumberUsingFromIsNotAllowed() =>
		Assert.That(() => ParseExpression("Number(\"5\")"),
			Throws.InstanceOf<Type.NoMatchingMethodFound>());

	[Test]
	public void ParseNumberToText()
	{
		var methodCall = (MethodCall)ParseExpression("5 to Text");
		Assert.That(methodCall.ReturnType.Name, Is.EqualTo(Base.Text));
		Assert.That(methodCall.Instance, Is.EqualTo(new Number(method, 5)));
	}

	private const string Case1 = "1";
	private const string Case2 = "7.59";
	private const string Case3 = "10";
	private const string Case4 = "0.5";
	private const string Case5 = "-50";
	private const string Case6 = "2000000102";
	private const string Case7 = "5045142575";
	private const string Case8 = "0";
	private const string Case9 = "7e-100";

	//ncrunch: no coverage start
	/// <summary>
	/// All benchmarks ran 1 million iterations, so normal usecase of numbers is 1 billions of a
	/// second, no text is a little less than that.
	/// SpanTryParseNumber				| 10,918.2 us | 328.83 us | 217.50 us |      1 KB |
	/// SpanTextTryParseNumber		|    935.3 us |  20.65 us |  13.66 us |      1 KB |
	/// SpanSimpleTryParseNumber	|  1,173.8 us |  39.75 us |  26.29 us |      1 KB |
	/// IntTryParse								| 30,177.0 us | 734.38 us | 485.75 us |      1 KB |
	/// DoubleTryParse						| 35,744.1 us | 639.56 us | 423.03 us |      1 KB |
	/// </summary>
	[Test]
	[Category("Manual")]
	[Benchmark]
	public void SpanTryParseNumber() =>
		Assert.That(SpanTryParseNumberPart1() + SpanTryParseNumberPart2(), Is.EqualTo(1000000));

	[Test]
	[Category("Manual")]
	[Benchmark]
	public void SpanTextTryParseNumber()
	{
		var counter = 0;
		var case10Span = NoNumberCase.AsSpan();
		for (var iteration = 0; iteration < 1000000; iteration++)
			if (!case10Span.TryParseNumber(out _))
				counter++;
		Assert.That(counter, Is.EqualTo(1000000));
	}

	[Test]
	[Category("Manual")]
	[Benchmark]
	public void SpanSimpleTryParseNumber()
	{
		var counter = 0;
		var case1Span = Case1.AsSpan();
		for (var iteration = 0; iteration < 1000000; iteration++)
			if (case1Span.TryParseNumber(out _))
				counter++;
		Assert.That(counter, Is.EqualTo(1000000));
	}

	private static int SpanTryParseNumberPart1()
	{
		var case1Span = Case1.AsSpan();
		var case2Span = Case2.AsSpan();
		var case3Span = Case3.AsSpan();
		var case4Span = Case4.AsSpan();
		var case5Span = Case5.AsSpan();
		var counter = 0;
		for (var iteration = 0; iteration < 100000; iteration++)
		{
			if (case1Span.TryParseNumber(out var result1) && result1 == 1)
				counter++;
			else
				throw new NotSupportedException(Case1);
			if (case2Span.TryParseNumber(out var result2) && result2 == 7.59)
				counter++;
			else
				throw new NotSupportedException(Case2);
			if (case3Span.TryParseNumber(out var result3) && result3 == 10)
				counter++;
			else
				throw new NotSupportedException(Case3);
			if (case4Span.TryParseNumber(out var result4) && result4 == 0.5)
				counter++;
			else
				throw new NotSupportedException(Case4);
			if (case5Span.TryParseNumber(out var result5) && result5 == -50)
				counter++;
			else
				throw new NotSupportedException(Case5);
		}
		return counter;
	}

	private static int SpanTryParseNumberPart2()
	{
		var case6Span = Case6.AsSpan();
		var case7Span = Case7.AsSpan();
		var case8Span = Case8.AsSpan();
		var case9Span = Case9.AsSpan();
		var case10Span = NoNumberCase.AsSpan();
		var counter = 0;
		for (var iteration = 0; iteration < 100000; iteration++)
		{
			if (case6Span.TryParseNumber(out var result6) && result6 == 2000000102)
				counter++;
			else
				throw new NotSupportedException(Case6);
			if (case7Span.TryParseNumber(out var result7) && result7 == 5045142575)
				counter++;
			else
				throw new NotSupportedException(Case7);
			if (case8Span.TryParseNumber(out var result8) && result8 == 0)
				counter++;
			else
				throw new NotSupportedException(Case8);
			if (case9Span.TryParseNumber(out var result9) && result9 == 7e-100)
				counter++;
			else
				throw new NotSupportedException(Case9);
			if (!case10Span.TryParseNumber(out _))
				counter++;
			else
				throw new NotSupportedException(NoNumberCase);
		}
		return counter;
	}

	private const string NoNumberCase = "text";

	[Category("Manual")]
	[Benchmark]
	[Test]
	public void IntTryParse()
	{
		var counter = 0;
		for (var iteration = 0; iteration < 100000; iteration++)
		{
			if (int.TryParse(Case1, out var result1) && result1 == 1)
				counter++;
			else
				throw new NotSupportedException(Case1);
			if (!int.TryParse(Case2, out _) && double.TryParse(Case2, out var result2) && result2 == 7.59)
				counter++;
			else
				throw new NotSupportedException(Case2);
			if (int.TryParse(Case3, out var result3) && result3 == 10)
				counter++;
			else
				throw new NotSupportedException(Case3);
			if (!int.TryParse(Case4, out _) && double.TryParse(Case4, out var result4) && result4 == 0.5)
				counter++;
			else
				throw new NotSupportedException(Case4);
			if (int.TryParse(Case5, out var result5) && result5 == -50)
				counter++;
			else
				throw new NotSupportedException(Case5);
			if (int.TryParse(Case6, out var result6) && result6 == 2000000102)
				counter++;
			else
				throw new NotSupportedException(Case6);
			if (!int.TryParse(Case7, out _) && double.TryParse(Case7, out var result7) &&
				result7 == 5045142575)
				counter++;
			else
				throw new NotSupportedException(Case7);
			if (int.TryParse(Case8.AsSpan(), out var result8) && result8 == 0)
				counter++;
			else
				throw new NotSupportedException(Case8);
			if (!int.TryParse(Case9, out _) && double.TryParse(Case9, out var result9) &&
				result9 == 7e-100)
				counter++;
			else
				throw new NotSupportedException(Case9);
			if (!int.TryParse(NoNumberCase, out _) && !double.TryParse(NoNumberCase, out _))
				counter++;
			else
				throw new NotSupportedException(NoNumberCase);
		}
		Assert.That(counter, Is.EqualTo(1000000));
	}

	[Category("Manual")]
	[Benchmark]
	[Test]
	public void DoubleTryParse()
	{
		var counter = 0;
		for (var iteration = 0; iteration < 100000; iteration++)
		{
			if (double.TryParse(Case1, out var result1) && result1 == 1)
				counter++;
			else
				throw new NotSupportedException(Case1);
			if (double.TryParse(Case2, out var result2) && result2 == 7.59)
				counter++;
			else
				throw new NotSupportedException(Case2);
			if (double.TryParse(Case3, out var result3) && result3 == 10)
				counter++;
			else
				throw new NotSupportedException(Case3);
			if (double.TryParse(Case4, out var result4) && result4 == 0.5)
				counter++;
			else
				throw new NotSupportedException(Case4);
			if (double.TryParse(Case5, out var result5) && result5 == -50)
				counter++;
			else
				throw new NotSupportedException(Case5);
			if (double.TryParse(Case6, out var result6) && result6 == 2000000102)
				counter++;
			else
				throw new NotSupportedException(Case6);
			if (double.TryParse(Case7, out var result7) && result7 == 5045142575)
				counter++;
			else
				throw new NotSupportedException(Case7);
			if (double.TryParse(Case8.AsSpan(), out var result8) && result8 == 0)
				counter++;
			else
				throw new NotSupportedException(Case8);
			if (double.TryParse(Case9, out var result9) && result9 == 7e-100)
				counter++;
			else
				throw new NotSupportedException(Case9);
			if (!double.TryParse(NoNumberCase, out _))
				counter++;
			else
				throw new NotSupportedException(NoNumberCase);
		}
		Assert.That(counter, Is.EqualTo(1000000));
	}

	[Category("Manual")]
	[Test]
	public void BenchmarkTryParseCompare() => BenchmarkRunner.Run<NumberTests>();
}