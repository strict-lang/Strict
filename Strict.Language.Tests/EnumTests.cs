using NUnit.Framework;
using Strict.Language.Expressions;

namespace Strict.Language.Tests;

public sealed class EnumTests
{
	[TestCase(true, "has log", "has number")]
	[TestCase(true, "has log")]
	[TestCase(false, "implement Number", "has log")]
	[TestCase(false, "has log", "Run", "\t5")]
	public void CheckTypeIsEnum(bool expected, params string[] lines)
	{
		var type = new Type(new TestPackage(),
			new TypeLines(nameof(CheckTypeIsEnum), lines)).ParseMembersAndMethods(new MethodExpressionParser());
		Assert.That(type.IsEnum, Is.EqualTo(expected));
	}
}