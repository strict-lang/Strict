using NUnit.Framework;
using Strict.Language.Expressions;

namespace Strict.Language.Tests;

public sealed class EnumTests
{
	[SetUp]
	public void CreatePackageAndParser()
	{
		package = new TestPackage();
		parser = new MethodExpressionParser();
		new Type(package,
			new TypeLines("Connection", "has Google = \"https://google.com\"",
				"has Apple = \"https://apple.com\"")).ParseMembersAndMethods(parser);
	}

	private Package package = null!;
	private ExpressionParser parser = null!;

	[TestCase(true, "has log", "has number")]
	[TestCase(true, "has log")]
	[TestCase(false, "implement Number", "has log")]
	[TestCase(false, "has log", "Run", "\t5")]
	public void CheckTypeIsEnum(bool expected, params string[] lines)
	{
		var type = new Type(package,
			new TypeLines(nameof(CheckTypeIsEnum), lines)).ParseMembersAndMethods(parser);
		Assert.That(type.IsEnum, Is.EqualTo(expected));
	}

	[Test]
	public void UseEnumWithoutConstructor()
	{
		var consumingType = new Type(package,
			new TypeLines(nameof(UseEnumWithoutConstructor), "has log",
				"Run", "\tlet url = Connection.Google")).ParseMembersAndMethods(parser);
		var assignment = (Assignment)consumingType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(assignment.Value, Is.InstanceOf<MemberCall>()!);
		var member = ((MemberCall)assignment.Value).Member;
		Assert.That(member.Name, Is.EqualTo("Google"));
		Assert.That(member.Type.Name, Is.EqualTo("Text"));
	}

	[Test]
	public void UseEnumAsMemberWithoutConstructor()
	{
		var consumingType = new Type(package,
			new TypeLines(nameof(UseEnumWithoutConstructor), "has url = Connection.Google",
				"Run", "\t5")).ParseMembersAndMethods(parser);
		Assert.That(consumingType.Members[0].Value, Is.InstanceOf<MemberCall>());
		var memberCall = (MemberCall)consumingType.Members[0].Value!;
		Assert.That(memberCall.Member.Name, Is.EqualTo("Google"));
		Assert.That(consumingType.Members[0].Type.Name, Is.EqualTo("Text"));
	}
}