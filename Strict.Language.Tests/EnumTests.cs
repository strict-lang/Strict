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
				"has Microsoft = \"https://microsoft.com\"")).ParseMembersAndMethods(parser);
		CreateInstructionEnumType();
	}

	private void CreateInstructionEnumType() =>
		new Type(package,
			new TypeLines("Instruction", "has Number", "has Set Number", "has Add Number", "has Subtract Number",
				"has Multiply Number", "has Divide Number", "has BinaryOperatorsSeparator = 100",
				"has GreaterThan Number", "has LessThan Number", "has Equal Number", "has NotEqual Number",
				"has ConditionalSeparator = 200", "has JumpIfTrue Number", "has JumpIfFalse Number",
				"has JumpIfNotZero Number", "has JumpsSeparator = 300")).ParseMembersAndMethods(parser);

	private Package package = null!;
	private ExpressionParser parser = null!;

	[TestCase(true, "has log", "has number")]
	[TestCase(true, "has log", "has boolean")]
	[TestCase(false, "has log", "Run", "\t5")]
	public void CheckTypeIsEnum(bool expected, params string[] lines)
	{
		var type = new Type(package,
			new TypeLines(nameof(CheckTypeIsEnum), lines)).ParseMembersAndMethods(parser);
		Assert.That(type.IsDatatypeOrEnum, Is.EqualTo(expected));
	}

	[Test]
	public void UseEnumWithoutConstructor()
	{
		var consumingType = new Type(package,
			new TypeLines(nameof(UseEnumWithoutConstructor), "has log",
				"Run", "\tconstant url = Connection.Google")).ParseMembersAndMethods(parser);
		var assignment = (ConstantDeclaration)consumingType.Methods[0].GetBodyAndParseIfNeeded();
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

	[Test]
	public void EnumWithoutValuesUsedAsMemberAndVariable()
	{
		var consumingType = new Type(package,
				new TypeLines(nameof(EnumWithoutValuesUsedAsMemberAndVariable),
					"has something = Instruction.Add", "Run", "\tconstant myInstruction = Instruction.Set")).
			ParseMembersAndMethods(parser);
		Assert.That(consumingType.GetType("Instruction").IsDatatypeOrEnum, Is.True);
		Assert.That(((MemberCall)consumingType.Members[0].Value!).Member.Name, Is.EqualTo("Add"));
		var assignment = (ConstantDeclaration)consumingType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(assignment.Value, Is.InstanceOf<MemberCall>()!);
		var member = ((MemberCall)assignment.Value).Member;
		Assert.That(member.Name, Is.EqualTo("Set"));
		Assert.That(member.Type.Name, Is.EqualTo("Number"));
	}

	[Test]
	public void CompareEnums()
	{
		var consumingType = new Type(package,
				new TypeLines(nameof(CompareEnums),
					"has receivedInstruction = Instruction.Add",
					"ExecuteInstruction(numbers) Number",
					"\tif receivedInstruction is Instruction.Add",
					"\t\treturn numbers(0) + numbers(1)")).
			ParseMembersAndMethods(parser);
		var ifExpression = (If)consumingType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(ifExpression.Condition.ReturnType.Name, Is.EqualTo(Base.Boolean));
		var binary = (Binary)ifExpression.Condition;
		Assert.That(((MemberCall)binary.Arguments[0]).Member.Name, Is.EqualTo("Add"));
		Assert.That(((MemberCall)((MemberCall)binary.Instance!).Member.Value!).Member.Name,
			Is.EqualTo("Add"));
	}

	[Test]
	public void UseEnumAsMethodParameters()
	{
		var consumingType = new Type(package,
				new TypeLines(nameof(UseEnumAsMethodParameters),
					"has log",
					"ExecuteInstruction(numbers, instruction) Number",
					"\tif instruction is Instruction.Add",
					"\t\treturn numbers(0) + numbers(1)",
					"CallExecute Number",
					"\tconstant result = ExecuteInstruction((1, 2), Instruction.Add)")).
			ParseMembersAndMethods(parser);
		var _ = (If)consumingType.Methods[0].GetBodyAndParseIfNeeded();
		var result = (ConstantDeclaration)consumingType.Methods[1].GetBodyAndParseIfNeeded();
		Assert.That(result.Value, Is.InstanceOf<MethodCall>());
		Assert.That(((MemberCall)((MethodCall)result.Value).Arguments[1]).Member.Name,
			Is.EqualTo("Add"));
	}
}