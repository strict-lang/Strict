using System;
using BenchmarkDotNet.Attributes;
using NUnit.Framework;

namespace Strict.Language.Tests;

public class BenchmarkTests
{
	[Benchmark]
	[TestCase("+")]
	[TestCase("*")]
	[TestCase("/")]
	[TestCase("-")]
	[TestCase(" randomwasdasdasdadasd +fewgfewrgfergergsdasda/dsa * ")]
	public void IsOperatorUsingStringContains(string code)
	{
		for (var index = 0; index < 1000000; index++)
			code.IsOperator();
	}

	[Benchmark]
	[TestCase("+")]
	[TestCase("*")]
	[TestCase("/")]
	[TestCase("-")]
	[TestCase(" randomwasdasdasdadasd +fewgfewrgfergergsdasda/dsa * ")]
	public void IsOperatorUsingReadOnlyMemory(string code)
	{
		for (var index = 0; index < 100000; index++)
			code.AsMemory().IsOperator();
	}
}