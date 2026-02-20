using System.Globalization;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Running;

namespace Strict.Expressions.Tests;

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
		Assert.That(ParseExpression(input),
			Is.EqualTo(new Number(method, double.Parse(input, CultureInfo.InvariantCulture))));

	private const string Case1 = "1";
	private const string Case2 = "7.59";
	private const string Case3 = "10";
	private const string Case4 = "0.5";
	private const string Case5 = "-50";
	private const string Case6 = "2000000102";
	private const string Case7 = "5045142575";
	private const string Case8 = "0";
	private const string Case9 = "7e-100";

	[Test]
	public void ParseTextToNumberUsingFromIsNotAllowed() =>
		Assert.That(() => ParseExpression("Number(\"5\")"),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.ArgumentsDoNotMatchMethodParameters>());

	[Test]
	public void ParseNumberToText()
	{
		var methodCall = (MethodCall)ParseExpression("5 to Text");
		Assert.That(methodCall.ReturnType.Name, Is.EqualTo(Base.Text));
		Assert.That(methodCall.Instance, Is.EqualTo(new Number(method, 5)));
	}

	[TestCase(Case1)]
	[TestCase(Case2)]
	[TestCase(Case3)]
	[TestCase(Case4)]
	[TestCase(Case7)]
	public void ParsingNumberShouldAlwaysResultInTheSameOutputText(string input) =>
		Assert.That(ParseExpression(input).ToString(), Is.EqualTo(input));

	[Test]
	public void CompareNumber()
	{
		Assert.That(ParseExpression(Case3), Is.EqualTo(ParseExpression(Case3)));
		Assert.That(ParseExpression(Case3), Is.EqualTo(new Number(type, 10)));
	}

	//ncrunch: no coverage start
	/// <summary>
	/// All benchmarks ran 1 million iterations, so normal usecase of numbers is 1 billion's of a
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
	public void SpanTryParseNumber()
	{
		SpanTryParseNumberCase(Case1, 1);
		SpanTryParseNumberCase(Case2, 7.59);
		SpanTryParseNumberCase(Case3, 10);
		SpanTryParseNumberCase(Case4, 0.5);
		SpanTryParseNumberCase(Case5, -50);
		SpanTryParseNumberCase(Case6, 2000000102);
		SpanTryParseNumberCase(Case7, 5045142575);
		SpanTryParseNumberCase(Case8, 0);
		SpanTryParseNumberCase(Case9, 7e-100);
		SpanTryParseNoNumberCase();
	}

	private static void SpanTryParseNumberCase(string caseText, double expected)
	{
		var counter = 0;
		var caseSpan = caseText.AsSpan();
		for (var iteration = 0; iteration < NumberOfIterations; iteration++)
			counter += caseSpan.TryParseNumber(out var result) && result == expected
				? 1
				: throw new NotSupportedException(caseText);
		Assert.That(counter, Is.EqualTo(NumberOfIterations));
	}

	private static void SpanTryParseNoNumberCase()
	{
		var counter = 0;
		var caseSpan = NoNumberCase.AsSpan();
		for (var iteration = 0; iteration < NumberOfIterations; iteration++)
			counter += !caseSpan.TryParseNumber(out _)
				? 1
				: throw new NotSupportedException(NoNumberCase);
		Assert.That(counter, Is.EqualTo(NumberOfIterations));
	}

	[Test]
	[Benchmark]
	public void SpanTextTryParseNumber()
	{
		var counter = 0;
		var case10Span = NoNumberCase.AsSpan();
		for (var iteration = 0; iteration < NumberOfIterations; iteration++)
			if (!case10Span.TryParseNumber(out _))
				counter++;
		Assert.That(counter, Is.EqualTo(NumberOfIterations));
	}

	[Test]
	[Benchmark]
	public void SpanSimpleTryParseNumber()
	{
		var counter = 0;
		var case1Span = Case1.AsSpan();
		for (var iteration = 0; iteration < NumberOfIterations; iteration++)
			if (case1Span.TryParseNumber(out _))
				counter++;
		Assert.That(counter, Is.EqualTo(NumberOfIterations));
	}

	private const string NoNumberCase = "text";

	[Test]
	[Benchmark]
	public void IntTryParse()
	{
		var counter = 0;
		for (var iteration = 0; iteration < NumberOfIterations; iteration++)
			counter += IntTryParseCase1To5() + IntTryParseCase6To10();
		Assert.That(counter, Is.EqualTo(NumberOfCases * NumberOfIterations));
	}

	private const int NumberOfCases = 10;
	private const int NumberOfIterations = 100000;

	private static int IntTryParseCase1To5() =>
		(int.TryParse(Case1, out var result1) && result1 == 1
			? 1
			: throw new NotSupportedException(Case1)) +
		(!int.TryParse(Case2, out _) &&
			double.TryParse(Case2, NumberFormatInfo.InvariantInfo, out var result2) && result2 == 7.59
				? 1
				: throw new NotSupportedException(Case2)) +
		(int.TryParse(Case3, out var result3) && result3 == 10
			? 1
			: throw new NotSupportedException(Case3)) +
		(!int.TryParse(Case4, out _) &&
			double.TryParse(Case4, NumberFormatInfo.InvariantInfo, out var result4) && result4 == 0.5
				? 1
				: throw new NotSupportedException(Case4)) +
		(int.TryParse(Case5, out var result5) && result5 == -50
			? 1
			: throw new NotSupportedException(Case5));

	private static int IntTryParseCase6To10() =>
		(int.TryParse(Case6, out var result6) && result6 == 2000000102
			? 1
			: throw new NotSupportedException(Case6)) +
		(!int.TryParse(Case7, out _) &&
			double.TryParse(Case7, out var result7) && result7 == 5045142575
				? 1
				: throw new NotSupportedException(Case7)) +
		(int.TryParse(Case8.AsSpan(), out var result8) && result8 == 0
			? 1
			: throw new NotSupportedException(Case8)) +
		(!int.TryParse(Case9, out _) &&
			double.TryParse(Case9, out var result9) && result9 == 7e-100
				? 1
				: throw new NotSupportedException(Case9)) +
		(!int.TryParse(NoNumberCase, out _) &&
			!double.TryParse(NoNumberCase, out _)
				? 1
				: throw new NotSupportedException(NoNumberCase));

	[Test]
	[Benchmark]
	public void DoubleTryParse()
	{
		var counter = 0;
		for (var iteration = 0; iteration < NumberOfIterations; iteration++)
			counter += DoubleTryParseCase1To5() + DoubleTryParseCase6To10();
		Assert.That(counter, Is.EqualTo(NumberOfCases * NumberOfIterations));
	}

	private static int DoubleTryParseCase1To5() =>
		(double.TryParse(Case5, NumberFormatInfo.InvariantInfo, out var result5) && result5 == -50
			? 1
			: throw new NotSupportedException(Case5)) +
		(double.TryParse(Case4, NumberFormatInfo.InvariantInfo, out var result4) && result4 == 0.5
			? 1
			: throw new NotSupportedException(Case4)) +
		(double.TryParse(Case3, NumberFormatInfo.InvariantInfo, out var result3) && result3 == 10
			? 1
			: throw new NotSupportedException(Case3)) +
		(double.TryParse(Case2, NumberFormatInfo.InvariantInfo, out var result2) && result2 == 7.59
			? 1
			: throw new NotSupportedException(Case2)) +
		(double.TryParse(Case1, NumberFormatInfo.InvariantInfo, out var result1) && result1 == 1
			? 1
			: throw new NotSupportedException(Case1));

	private static int DoubleTryParseCase6To10() =>
		(double.TryParse(Case6, out var result6) && result6 == 2000000102
			? 1
			: throw new NotSupportedException(Case6)) +
		(double.TryParse(Case7, out var result7) && result7 == 5045142575
			? 1
			: throw new NotSupportedException(Case7)) +
		(double.TryParse(Case8.AsSpan(), out var result8) && result8 == 0
			? 1
			: throw new NotSupportedException(Case8)) +
		(double.TryParse(Case9, out var result9) && result9 == 7e-100
			? 1
			: throw new NotSupportedException(Case9)) +
		(!double.TryParse(NoNumberCase, out _)
			? 1
			: throw new NotSupportedException(NoNumberCase));

	[Test]
	[Category("Manual")]
	public void BenchmarkTryParseCompare() => BenchmarkRunner.Run<NumberTests>();
}