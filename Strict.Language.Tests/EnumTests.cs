namespace Strict.Language.Tests;

public sealed class EnumTests
{
	[SetUp]
	public void CreatePackageAndParser()
	{
		parser = new MethodExpressionParser();
		connectionType = new Type(TestPackage.Instance,
			new TypeLines("Connection", "constant Google = \"https://google.com\"",
				"constant Microsoft = \"https://microsoft.com\"")).ParseMembersAndMethods(parser);
		instructionType = new Type(TestPackage.Instance,
			new TypeLines("Instruction", "constant Set", "constant Add", "constant Subtract",
				"constant Multiply", "constant Divide", "constant BinaryOperatorsSeparator = 100",
				"constant GreaterThan", "constant LessThan", "constant Equal", "constant NotEqual",
				"constant ConditionalSeparator", "constant JumpIfTrue", "constant JumpIfFalse",
				"constant JumpIfNotZero", "constant JumpsSeparator = 300")).ParseMembersAndMethods(parser);
	}

	private ExpressionParser parser = null!;
	private Type connectionType = null!;
	private Type instructionType = null!;

	[TearDown]
	public void TearDown()
	{
		TestPackage.Instance.Remove(connectionType);
		TestPackage.Instance.Remove(instructionType);
	}

	[TestCase(true, "constant Set", "constant Add")]
	[TestCase(false, "has logger", "has number")]
	[TestCase(false, "has logger", "constant Add")]
	[TestCase(false, "has logger", "Run", "\t5")]
	public void CheckTypeIsEnum(bool expected, params string[] lines)
	{
		using var type = new Type(TestPackage.Instance,
			new TypeLines(nameof(CheckTypeIsEnum), lines)).ParseMembersAndMethods(parser);
		Assert.That(type.IsEnum, Is.EqualTo(expected));
	}

	[Test]
	public void UseEnumWithoutConstructor()
	{
		var consumingType = new Type(TestPackage.Instance,
			new TypeLines(nameof(UseEnumWithoutConstructor), "has logger",
				"Run", "\tconstant url = Connection.Google")).ParseMembersAndMethods(parser);
		var assignment = (Declaration)consumingType.Methods[^1].GetBodyAndParseIfNeeded();
		Assert.That(assignment.Value, Is.InstanceOf<MemberCall>());
		var member = ((MemberCall)assignment.Value).Member;
		Assert.That(member.Name, Is.EqualTo("Google"));
		Assert.That(member.Type.Name, Is.EqualTo("Text"));
	}

	[Test]
	public void UseEnumAsMemberWithoutConstructor()
	{
		var consumingType = new Type(TestPackage.Instance,
			new TypeLines(nameof(UseEnumAsMemberWithoutConstructor), "constant url = Connection.Google",
				"Run", "\t5")).ParseMembersAndMethods(parser);
		Assert.That(consumingType.Members[0].InitialValue, Is.InstanceOf<MemberCall>());
		var memberCall = (MemberCall)consumingType.Members[0].InitialValue!;
		Assert.That(memberCall.Member.Name, Is.EqualTo("Google"));
		Assert.That(consumingType.Members[0].Type.Name, Is.EqualTo("Text"));
	}

	[Test]
	public void EnumWithoutValuesUsedAsMemberAndVariable()
	{
		var consumingType = new Type(TestPackage.Instance,
				new TypeLines(nameof(EnumWithoutValuesUsedAsMemberAndVariable),
					"has something = Instruction.Add", "Run", "\tconstant myInstruction = Instruction.Set")).
			ParseMembersAndMethods(parser);
		Assert.That(consumingType.GetType("Instruction").IsEnum, Is.True);
		Assert.That(((MemberCall)consumingType.Members[0].InitialValue!).Member.Name, Is.EqualTo("Add"));
		var assignment = (Declaration)consumingType.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(assignment.Value, Is.InstanceOf<MemberCall>());
		var member = ((MemberCall)assignment.Value).Member;
		Assert.That(member.Name, Is.EqualTo("Set"));
		Assert.That(member.Type.Name, Is.EqualTo("Number"));
	}

	[Test]
	public void CompareEnums()
	{
		var consumingType = new Type(TestPackage.Instance,
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
		var consumingType = new Type(TestPackage.Instance,
				new TypeLines(nameof(UseEnumAsMethodParameters),
					"has logger",
					"ExecuteInstruction(numbers, instruction) Number",
					"\tif instruction is Instruction.Add",
					"\t\treturn numbers(0) + numbers(1)",
					"CallExecute Number",
					"\tconstant result = ExecuteInstruction((1, 2), Instruction.Add)")).
			ParseMembersAndMethods(parser);
		consumingType.Methods[0].GetBodyAndParseIfNeeded();
		var result = (Declaration)consumingType.Methods[1].GetBodyAndParseIfNeeded();
		Assert.That(result.Value, Is.InstanceOf<MethodCall>());
		Assert.That(((MemberCall)((MethodCall)result.Value).Arguments[1]).Member.Name,
			Is.EqualTo("Add"));
	}

	[Test]
	public void EnumCanHaveMembersWithDifferentTypes()
	{
		var type = new Type(TestPackage.Instance,
				new TypeLines(nameof(EnumCanHaveMembersWithDifferentTypes),
					"constant One",
					"constant SomeText = \"2\"",
					"constant InputFile = File(\"test.txt\")")).
			ParseMembersAndMethods(parser);
		Assert.That(type.IsEnum, Is.EqualTo(true));
	}

	[Test]
	public void UseEnumExtensions()
	{
		new Type(TestPackage.Instance,
			new TypeLines("MoreInstruction", "has instruction", "constant BlaDivide = 14",
				"constant BlaBinaryOperatorsSeparator", "constant BlaGreaterThan",
				"constant BlaLessThan")).ParseMembersAndMethods(parser);
		var body = (Body)new Type(TestPackage.Instance,
				new TypeLines(nameof(UseEnumExtensions), "has logger", "UseExtendedEnum(instruction) Number",
					"\tlet result = instruction to MoreInstruction", "\tresult.BlaDivide")).
			ParseMembersAndMethods(parser).Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.Expressions[1].ToString(), Is.EqualTo("result.BlaDivide"));
	}

	[Test]
	public void EnumConstantsCanBeUsedDirectly()
	{
		var type =
			new Type(TestPackage.Instance,
				new TypeLines(nameof(EnumConstantsCanBeUsedDirectly), "has instruction", "Run",
					"\tInstruction.Multiply is not instruction")).ParseMembersAndMethods(parser);
		var method = type.FindMethod("Run", []);
		Assert.That(method, Is.Not.Null);
		var call = (MethodCall)method!.GetBodyAndParseIfNeeded();
		Assert.That(call.ToString(), Is.EqualTo("Instruction.Multiply is not instruction"));
	}
}