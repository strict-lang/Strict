using NUnit.Framework;
using Strict.Expressions;

namespace Strict.Language.Tests;

public sealed class EnumTests
{
	[SetUp]
	public void CreatePackageAndParser()
	{
		package = new TestPackage();
		parser = new MethodExpressionParser();
		new Type(package,
			new TypeLines("Connection", "constant Google = \"https://google.com\"",
				"constant Microsoft = \"https://microsoft.com\"")).ParseMembersAndMethods(parser);
		new Type(package,
			new TypeLines("Instruction", "constant Set", "constant Add", "constant Subtract",
				"constant Multiply", "constant Divide", "constant BinaryOperatorsSeparator = 100",
				"constant GreaterThan", "constant LessThan", "constant Equal", "constant NotEqual",
				"constant ConditionalSeparator", "constant JumpIfTrue", "constant JumpIfFalse",
				"constant JumpIfNotZero", "constant JumpsSeparator = 300")).ParseMembersAndMethods(parser);
	}

	private Package package = null!;
	private ExpressionParser parser = null!;

	[TestCase(true, "constant Set", "constant Add")]
	[TestCase(false, "has log", "has number")]
	[TestCase(false, "has log", "constant Add")]
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
				"Run", "\tconstant url = Connection.Google")).ParseMembersAndMethods(parser);
		var assignment = (ConstantDeclaration)consumingType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(assignment.Value, Is.InstanceOf<MemberCall>());
		var member = ((MemberCall)assignment.Value).Member;
		Assert.That(member.Name, Is.EqualTo("Google"));
		Assert.That(member.Type.Name, Is.EqualTo("Text"));
	}

	[Test]
	public void UseEnumAsMemberWithoutConstructor()
	{
		var consumingType = new Type(package,
			new TypeLines(nameof(UseEnumWithoutConstructor), "constant url = Connection.Google",
				"Run", "\t5")).ParseMembersAndMethods(parser);
		Assert.That(consumingType.Members[0].InitialValue, Is.InstanceOf<MemberCall>());
		var memberCall = (MemberCall)consumingType.Members[0].InitialValue!;
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
		Assert.That(consumingType.GetType("Instruction").IsEnum, Is.True);
		Assert.That(((MemberCall)consumingType.Members[0].InitialValue!).Member.Name, Is.EqualTo("Add"));
		var assignment = (ConstantDeclaration)consumingType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(assignment.Value, Is.InstanceOf<MemberCall>());
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
		Assert.That(((MemberCall)((MemberCall)binary.Instance!).Member.InitialValue!).Member.Name,
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
		consumingType.Methods[0].GetBodyAndParseIfNeeded();
		var result = (ConstantDeclaration)consumingType.Methods[1].GetBodyAndParseIfNeeded();
		Assert.That(result.Value, Is.InstanceOf<MethodCall>());
		Assert.That(((MemberCall)((MethodCall)result.Value).Arguments[1]).Member.Name,
			Is.EqualTo("Add"));
	}

	[Test]
	public void EnumCanHaveMembersWithDifferentTypes()
	{
		var type = new Type(package,
				new TypeLines(nameof(EnumCanHaveMembersWithDifferentTypes),
					"constant One",
					"constant SomeText = \"2\"",
					"constant InputFile = File(\"test.txt\")")).
			ParseMembersAndMethods(parser);
		Assert.That(type.IsEnum, Is.EqualTo(true));
	}
	/*TODO: not sure if this is useful or not … not really supported to mix has instruction with rest enum constants.
	[Test]
	public void UseEnumExtensions()
	{
		new Type(package,
			new TypeLines("MoreInstruction", "has instruction", "constant BlaDivide = 14",
				"constant BlaBinaryOperatorsSeparator", "constant BlaGreaterThan",
				"constant BlaLessThan")).ParseMembersAndMethods(parser);
		var body = (Body)new Type(package,
				new TypeLines(nameof(UseEnumExtensions), "has log", "UseExtendedEnum(instruction) Number",
					"\tconstant result = instruction to MoreInstruction", "\tresult.BlaDivide")).
			ParseMembersAndMethods(parser).Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.Expressions[1].ToString(), Is.EqualTo("result.BlaDivide"));
	}
	*/
}