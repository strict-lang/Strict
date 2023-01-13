using System;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using Strict.Language.Expressions;
using static Strict.Language.Body;
using List = Strict.Language.Expressions.List;

namespace Strict.Language.Tests;

// ReSharper disable once ClassTooBig
public sealed class TypeTests
{
	[SetUp]
	public void CreatePackage()
	{
		package = new TestPackage();
		parser = new MethodExpressionParser();
		CreateType(Base.App, "Run");
	}

	private Type CreateType(string name, params string[] lines) =>
		new Type(package, new TypeLines(name, lines)).ParseMembersAndMethods(parser);

	public Package package = null!;
	public ExpressionParser parser = null!;

	[Test]
	public void AddingTheSameNameIsNotAllowed() =>
		Assert.That(() => CreateType(Base.App, "Run"),
			Throws.InstanceOf<Type.TypeAlreadyExistsInPackage>());

	[Test]
	public void EmptyLineIsNotAllowed() =>
		Assert.That(() => CreateType(Base.HashCode, ""),
			Throws.InstanceOf<Type.EmptyLineIsNotAllowed>().With.Message.Contains("line 1"));

	[Test]
	public void WhitespacesAreNotAllowed()
	{
		Assert.That(() => CreateType("Whitespace", " "),
			Throws.InstanceOf<Type.ExtraWhitespacesFoundAtBeginningOfLine>());
		Assert.That(() => CreateType("Program", " has App"),
			Throws.InstanceOf<Type.ExtraWhitespacesFoundAtBeginningOfLine>());
		Assert.That(() => CreateType(Base.HashCode, "has\t"),
			Throws.InstanceOf<Type.ExtraWhitespacesFoundAtEndOfLine>());
	}

	[Test]
	public void TypeParsersMustStartWithMember() =>
		Assert.That(() => CreateType(Base.HashCode, "Run", "\tlog.WriteLine"),
			Throws.InstanceOf<Type.TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies>());

	[Test]
	public void JustMembersAreAllowed() =>
		Assert.That(CreateType(Base.HashCode, "has log", "mutable counter Number").Members.Count, Is.EqualTo(2));

	[Test]
	public void GetUnknownTypeWillCrash() =>
		Assert.That(() => package.GetType(Base.Computation),
			Throws.InstanceOf<Context.TypeNotFound>());

	[TestCase("has invalidType")]
	[TestCase("has log", "Run InvalidType", "\tconstant a = 5")]
	public void TypeNotFound(params string[] lines) =>
		Assert.That(() => CreateType(Base.HashCode, lines),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.InstanceOf<Context.TypeNotFound>());

	[Test]
	public void NoMethodsFound() =>
		Assert.That(
			() => new Type(new Package(nameof(NoMethodsFound)), new TypeLines("dummy", "has Number")).
				ParseMembersAndMethods(null!), Throws.InstanceOf<Type.NoMethodsFound>());

	[Test]
	public void ExtraWhitespacesFoundAtBeginningOfLine() =>
		Assert.That(
			() => CreateType(nameof(ExtraWhitespacesFoundAtBeginningOfLine), "has log", "Run",
				" constant a = 5"), Throws.InstanceOf<Type.ExtraWhitespacesFoundAtBeginningOfLine>());

	[Test]
	public void NoMatchingMethodFound() =>
		Assert.That(
			() => CreateType(nameof(NoMatchingMethodFound), "has log", "Run", "\tconstant a = 5").
				GetMethod("UnknownMethod", Array.Empty<Expression>()),
			Throws.InstanceOf<Type.NoMatchingMethodFound>());

	[Test]
	public void TypeNameMustBeWord() =>
		Assert.That(() => new Member(package.GetType(Base.App), "blub7", null!),
			Throws.InstanceOf<Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers>());

	[TestCase("has any")]
	[TestCase("has random Any")]
	public void MemberWithTypeAnyIsNotAllowed(string line) =>
		Assert.That(() => CreateType("Program", line),
			Throws.InstanceOf<Type.MemberWithTypeAnyIsNotAllowed>());

	[TestCase("has log", "Run", "\tconstant result = Any")]
	[TestCase("has log", "Run", "\tconstant result = Any(5)")]
	[TestCase("has log", "Run", "\tconstant result = 5 + Any(5)")]
	public void VariableWithTypeAnyIsNotAllowed(params string[] lines)
	{
		var type = new Type(package, new TypeLines(nameof(VariableWithTypeAnyIsNotAllowed), lines)).ParseMembersAndMethods(parser);
		Assert.That(() => type.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<MethodExpressionParser.ExpressionWithTypeAnyIsNotAllowed>().With.Message.
				Contains("Any"));
	}

	[TestCase("has log", "Run(any)", "\tconstant result = 5")]
	[TestCase("has log", "Run(input Any)", "\tconstant result = 5")]
	public void MethodParameterWithTypeAnyIsNotAllowed(params string[] lines) =>
		Assert.That(() => CreateType("Program", lines),
			Throws.InstanceOf<Method.ParametersWithTypeAnyIsNotAllowed>());

	[Test]
	public void MethodReturnTypeAsAnyIsNotAllowed() =>
		Assert.That(() => CreateType("Program", "has log", "Run Any", "\tconstant result = 5"),
			Throws.InstanceOf<Method.MethodReturnTypeAsAnyIsNotAllowed>()!);

	[Test]
	public void MembersMustComeBeforeMethods() =>
		Assert.That(() => CreateType("Program", "Run", "has log"),
			Throws.InstanceOf<Type.MembersMustComeBeforeMethods>());

	[Test]
	public void SimpleApp() =>
		// @formatter:off
		CheckApp(CreateType("Program",
			"has App",
			"has log",
			"Run",
			"\tlog.Write(\"Hello World!\")"));

	private static void CheckApp(Type program)
	{
		Assert.That(program.Members[0].Type.Name, Is.EqualTo(Base.App));
		Assert.That(program.Members[1].Name, Is.EqualTo("log"));
		Assert.That(program.Methods[0].Name, Is.EqualTo("Run"));
		Assert.That(program.IsTrait, Is.False);
	}

	[Test]
	public void AnotherApp() =>
		CheckApp(CreateType("Program",
			"has App",
			"has log",
			"Run",
			"\tfor number in Range(0, 10)",
			"\t\tlog.Write(\"Counting: \" + number)"));

	[Test]
	public void NotImplementingAnyTraitMethodsAreAllowed() =>
		Assert.That(() => CreateType("Program",
				"has App",
				"add(number)",
				"\treturn one + 1"), Is.Not.Null);

	[Test]
	public void CannotImplementFewTraitMethodsAndLeaveOthers()
	{
		var type = new Type(package,
			new TypeLines(nameof(CannotImplementFewTraitMethodsAndLeaveOthers),
				"has file = \"test.txt\"", "Write(text)", "\tfile.Write(text)"));
		Assert.That(() => type.ParseMembersAndMethods(parser),
			Throws.InstanceOf<Type.MustImplementAllTraitMethodsOrNone>());
	}

	[Test]
	public void TraitMethodsMustBeImplemented() =>
		Assert.That(() => CreateType("Program",
				"has App",
				"Run"),
			Throws.InstanceOf<Type.MethodMustBeImplementedInNonTrait>());
	// @formatter:on

	[Test]
	public void Trait()
	{
		var app = CreateType("DummyApp", "Run");
		Assert.That(app.IsTrait, Is.True);
		Assert.That(app.Name, Is.EqualTo("DummyApp"));
		Assert.That(app.Methods[0].Name, Is.EqualTo("Run"));
	}

	[Test]
	public void CanUpCastNumberWithList()
	{
		var type = CreateType(nameof(CanUpCastNumberWithList), "has log",
			"Add(first Number, other Numbers) List", "\tfirst + other");
		var result = type.FindMethod("Add", new List<Expression>
		{
			new Number(type, 5),
			new List(null!, new List<Expression> { new Number(type, 6), new Number(type, 7) })
		});
		Assert.That(result, Is.InstanceOf<Method>());
		Assert.That(result?.ToString(),
			Is.EqualTo("Add(first TestPackage.Number, other TestPackage.Numbers) List"));
	}

	[Test]
	public void GenericTypesCannotBeUsedDirectlyUseImplementation()
	{
		var type = CreateType(nameof(GenericTypesCannotBeUsedDirectlyUseImplementation), "has generic",
			"AddGeneric(first Generic, other List) List", "\tfirst + other");
		Assert.That(
			() => type.FindMethod("AddGeneric",
				new List<Expression>
				{
					new Number(type, 6),
					new List(null!, new List<Expression> { new Number(type, 7), new Number(type, 8) })
				}), Throws.InstanceOf<Type.GenericTypesCannotBeUsedDirectlyUseImplementation>());
	}

	[TestCase(Base.Number, "has number", "Run", "\tmutable result = 2")]
	[TestCase(Base.Text, "has number", "Run", "\tmutable result = \"2\"")]
	public void MutableTypesHaveProperDataReturnType(string expected, params string[] code)
	{
		var expression = (ConstantDeclaration)
			new Type(package, new TypeLines(nameof(MutableTypesHaveProperDataReturnType), code)).
				ParseMembersAndMethods(parser).Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(expression.Value.ReturnType.Name, Is.EqualTo(expected));
	}

	[TestCase("has number", "Run", "\tnumber = 1 + 1")]
	[TestCase("has number", "Run", "\tconstant result = 5", "\tresult = 6")]
	public void ImmutableTypesCannotBeChanged(params string[] code) =>
		Assert.That(
			() => new Type(package, new TypeLines(nameof(ImmutableTypesCannotBeChanged), code)).ParseMembersAndMethods(parser).Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<ValueIsNotMutableAndCannotBeChanged>());

	[TestCase("mutable canBeModified = 0", "Run", "\tcanBeModified = 5")]
	[TestCase("mutable counter = 0", "Run", "\tcounter = 5")]
	public void MutableMemberTypesCanBeChanged(params string[] code)
	{
		var type = new Type(package, new TypeLines(nameof(MutableMemberTypesCanBeChanged), code)).
			ParseMembersAndMethods(parser);
		type.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(type.Members[0].Value, Is.EqualTo(new Number(type, 5)));
	}

	[Test]
	public void MutableVariableCanBeChanged()
	{
		var type = new Type(package, new TypeLines(nameof(MutableVariableCanBeChanged), "has number",
				"Run",
				"\tmutable result = 2",
				"\tresult = 5")).
			ParseMembersAndMethods(parser);
		var body = (Body)type.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.FindVariableValue("result")!.ToString(), Is.EqualTo("5"));
	}

	[Test]
	public void ValueTypeNotMatchingWithAssignmentType() =>
		Assert.That(
			() => new Type(package,
					new TypeLines(nameof(ValueTypeNotMatchingWithAssignmentType), "has log", "Run",
						"\tlog.Write(5) = 6")).ParseMembersAndMethods(parser).Methods[0].
				GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<MutableAssignment.ValueTypeNotMatchingWithAssignmentType>());

	[Test]
	public void MakeSureGenericTypeIsProperlyGenerated()
	{
		var listType = package.GetType(Base.List);
		Assert.That(listType.IsGeneric, Is.True);
		Assert.That(listType.Members[0].Type, Is.EqualTo(package.GetType(Base.Iterator)));
		var getNumbersBody = new Type(package,
				new TypeLines(nameof(MakeSureGenericTypeIsProperlyGenerated), "has numbers", "GetNumbers Numbers",
					"\tnumbers")).ParseMembersAndMethods(parser).
			Methods[0].GetBodyAndParseIfNeeded();
		var numbersType = package.GetListImplementationType(package.GetType(Base.Number));
		Assert.That(getNumbersBody.ReturnType, Is.EqualTo(numbersType));
		Assert.That(numbersType.Generic, Is.EqualTo(package.GetType(Base.List)));
		Assert.That(numbersType.ImplementationTypes[0], Is.EqualTo(package.GetType(Base.Number)));
	}

	[Test]
	public void CannotGetGenericImplementationOnNonGenericType() =>
		Assert.That(
			() => package.GetType(Base.Text).
				GetGenericImplementation(new List<Type> { package.GetType(Base.Number) }),
			Throws.InstanceOf<Type.CannotGetGenericImplementationOnNonGeneric>());

	[Test]
	public void UsingGenericMethodIsAllowed()
	{
		var type = CreateType(nameof(CanUpCastNumberWithList), "has log",
			"Add(other Texts, first Generic) List", "\tother + first");
		Assert.That(
			type.FindMethod("Add",
				new List<Expression>
				{
					new List(null!,
						new List<Expression> { new Text(type, "Hi"), new Text(type, "Hello") }),
					new Number(type, 5)
				})?.ToString(),
			Is.EqualTo("Add(other TestPackage.Texts, first TestPackage.Generic) List"));
	}

	[Test]
	public void GenericMethodShouldAcceptAllInputTypes()
	{
		var type = CreateType(nameof(GenericMethodShouldAcceptAllInputTypes),
			"has Output",
			"has log",
			"Write(generic)", "\tlog.Write(generic)");
		Assert.That(
			type.FindMethod("Write",
				new List<Expression>
				{
					new Text(type, "hello")
				})?.ToString(),
			Is.EqualTo("Write(generic TestPackage.Generic)"));
		Assert.That(
			type.FindMethod("Write",
				new List<Expression>
				{
					new Number(type, 5)
				})?.ToString(),
			Is.EqualTo("Write(generic TestPackage.Generic)"));
	}

	[Test]
	public void NonGenericExpressionCannotBeGeneric() =>
		Assert.That(
			() => new Type(package,
					new TypeLines(nameof(NonGenericExpressionCannotBeGeneric),
						"has list",
						"Something",
						"\tconstant result = list + 5")).ParseMembersAndMethods(parser).
				Methods[0].GetBodyAndParseIfNeeded(), Throws.InstanceOf<Type.GenericTypesCannotBeUsedDirectlyUseImplementation>());

	[Test]
	public void InvalidProgram() =>
		Assert.That(
			() => new Type(package,
				new TypeLines(nameof(InvalidProgram),
					"has list",
					"Something41",
					"\tconstant result = list + 5")).ParseMembersAndMethods(null!),
			Throws.InstanceOf<ParsingFailed>());

	[Test]
	public void MethodParameterCanBeGeneric()
	{
		var type = new Type(package,
			new TypeLines(nameof(InvalidProgram), "has log", "Something(input Generics)",
				"\tconstant result = input + 5")).ParseMembersAndMethods(parser);
		Assert.That(
			type.FindMethod("Something",
				new List<Expression>
				{
					new List(null!, new List<Expression> { new Text(type, "hello") })
				}), Is.Not.Null);
	}

	[Test]
	public void CreateTypeUsingConstructorMembers()
	{
		new Type(package,
			new TypeLines("Customer", "has text", "has age Number", "Print Text",
				"\t\"Customer Name: \" + name + \" Age: \" + age")).ParseMembersAndMethods(parser);
		var createCustomer = new Type(package,
			new TypeLines(nameof(CreateTypeUsingConstructorMembers), "has log", "Something",
				"\tconstant customer = Customer(\"Murali\", 28)")).ParseMembersAndMethods(parser);
		var assignment = (ConstantDeclaration)createCustomer.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(assignment.Value.ReturnType.Name, Is.EqualTo("Customer"));
		Assert.That(assignment.Value.ToString(), Is.EqualTo("Customer(\"Murali\", 28)"));
	}

	[Test]
	public void UsingToMethodForComplexTypeConstructorIsForbidden()
	{
		new Type(package,
			new TypeLines("Customer", "has text", "has age Number", "Print Text",
				"\t\"Customer Name: \" + name + \" Age: \" + age")).ParseMembersAndMethods(parser);
		var createCustomer = new Type(package,
			new TypeLines(nameof(CreateTypeUsingConstructorMembers), "has log", "Something",
				"\tconstant customer = (\"Murali\", 28) to Customer")).ParseMembersAndMethods(parser);
		Assert.That(() => createCustomer.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<List.ListElementsMustHaveMatchingType>());
	}

	[Test]
	public void CreateStacktraceTypeUsingMembersInConstructor()
	{
		var logger = new Type(package,
			new TypeLines("Logger", "has log", "has method", "has trace = Stacktrace(method, \"filePath\", 5)", "Log Text",
				"\tlog.Write(stacktrace to Text)")).ParseMembersAndMethods(parser);
		Assert.That(logger.Members[2].Type.Name, Is.EqualTo("Stacktrace"));
		Assert.That(logger.Members[2].Type.Members.Count, Is.EqualTo(3));
	}

	[Test]
	public void MutableTypesOrImplementsShouldNotBeUsedDirectly()
	{
		var type = new Type(package, new TypeLines(nameof(MutableTypesOrImplementsShouldNotBeUsedDirectly), "has number",
				"Run",
				"\tmutable result = Mutable(2)")).
			ParseMembersAndMethods(parser);
		Assert.That(() => type.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Type.GenericTypesCannotBeUsedDirectlyUseImplementation>());
	}

	[Test]
	public void RangeTypeShouldHaveCorrectAvailableMethods()
	{
		var range = package.GetType(Base.Range);
		Assert.That(range.AvailableMethods.Values.Select(methods => methods.Count).Sum(),
			Is.EqualTo(7));
	}

	[Test]
	public void TextTypeShouldHaveCorrectAvailableMethods()
	{
		var text = package.GetType(Base.Text + "s");
		Assert.That(text.AvailableMethods.Values.Select(methods => methods.Count).Sum(),
			Is.EqualTo(69));
	}

	[Test]
	public void PrivateMethodsShouldNotBeAddedToAvailableMethods()
	{
		var type = new Type(package, new TypeLines(nameof(PrivateMethodsShouldNotBeAddedToAvailableMethods),
			"has output", "run", "\tconstant n = 5"));
		type.ParseMembersAndMethods(parser);
		Assert.That(type.Methods.Count, Is.EqualTo(1));
		Assert.That(type.AvailableMethods.Keys.Contains("run"), Is.False);
	}

	[Test]
	public void AvailableMethodsShouldNotHaveMembersPrivateMethods()
	{
		new Type(package,
			new TypeLines("ProgramWithPublicAndPrivateMethods", "has log", "PublicMethod", "\tlog.Write(\"I am exposed\")", "privateMethod", "\tlog.Write(\"Support privacy\")")).ParseMembersAndMethods(parser);
		var type = new Type(package,
			new TypeLines(nameof(AvailableMethodsShouldNotHaveMembersPrivateMethods),
				"has programWithPublicAndPrivateMethods", "run", "\tconstant n = 5"));
		type.ParseMembersAndMethods(parser);
		Assert.That(type.AvailableMethods.Keys.Contains("privateMethod"), Is.False);
		Assert.That(type.AvailableMethods.Keys.Contains("PublicMethod"), Is.True);
		Assert.That(
			type.AvailableMethods.Values.Select(methodsList => methodsList).Any(methods =>
				methods.Any(method => !method.IsPublic && !method.Name.AsSpan().IsOperator())), Is.False);
	}

	[TestCase(Base.Output, 0)]
	[TestCase(Base.Mutable, 1)]
	[TestCase(Base.Log, 1)]
	[TestCase(Base.Number, 1)]
	[TestCase(Base.Character, 2)]
	[TestCase(Base.Text, 4)]
	[TestCase(Base.Error, 10)]
	public void ValidateAvailableMemberTypesCount(string name, int expectedCount)
	{
		var type = package.GetType(name);
		Assert.That(type.AvailableMemberTypes.Count, Is.EqualTo(expectedCount),
			type.AvailableMemberTypes.ToWordList());
	}

	[TestCase("has numbers with Length is 2")]
	[TestCase("has something Numbers with Length is 2")]
	public void MemberWithConstraintsUsingWithKeyword(string code)
	{
		var memberWithConstraintType = CreateType(nameof(MemberWithConstraintsUsingWithKeyword), code,
			"AddNumbers Number", "\tnumbers(0) + numbers(1)");
		var member = memberWithConstraintType.Members[0];
		Assert.That(member.Type.Name, Is.EqualTo("Numbers"));
		Assert.That(member.Constraints?.Length, Is.EqualTo(1));
		Assert.That(member.Constraints?[0].ToString(), Is.EqualTo("Length is 2"));
	}

	[Test]
	public void MutableMemberWithConstraintsUsingWithKeyword()
	{
		var memberWithConstraintType = CreateType(nameof(MutableMemberWithConstraintsUsingWithKeyword), "mutable something with Length is 2 = (1, 2)",
			"AddNumbers Number", "\tnumbers(0) + numbers(1)");
		var member = memberWithConstraintType.Members[0];
		Assert.That(member.Name, Is.EqualTo("something"));
		Assert.That(member.Type.Name, Is.EqualTo("Numbers"));
		Assert.That(member.Constraints?.Length, Is.EqualTo(1));
		Assert.That(member.Constraints?[0].ToString(), Is.EqualTo("Length is 2"));
		Assert.That(member.Value, Is.InstanceOf<List>());
		Assert.That(member.Value?.ToString(), Is.EqualTo("(1, 2)"));
	}

	[Test]
	public void MemberWithMultipleConstraintsUsingAndKeyword()
	{
		var memberWithConstraintType = CreateType(nameof(MemberWithMultipleConstraintsUsingAndKeyword), "mutable numbers with Length is 2 and value(0) > 0",
			"AddNumbers Number", "\tnumbers(0) + numbers(1)");
		var member = memberWithConstraintType.Members[0];
		Assert.That(member.Name, Is.EqualTo("numbers"));
		Assert.That(member.Type.Name, Is.EqualTo("Numbers"));
		Assert.That(member.Constraints?.Length, Is.EqualTo(2));
		Assert.That(member.Constraints?[1].ToString(), Is.EqualTo("value(0) > 0"));
	}

	[Test]
	public void ConstraintsWithOtherThanBooleanReturnTypeIsInvalid() =>
		Assert.That(
			() => CreateType(nameof(ConstraintsWithOtherThanBooleanReturnTypeIsInvalid),
				"mutable numbers with Length + 2", "AddNumbers Number", "\tnumbers(0) + numbers(1)"),
			Throws.InstanceOf<Member.InvalidConstraintExpression>());

	[Test]
	public void MissingConstraintExpression() =>
		Assert.That(
			() => CreateType(nameof(MissingConstraintExpression),
				"mutable numbers with", "AddNumbers Number", "\tnumbers(0) + numbers(1)"),
			Throws.InstanceOf<Type.MemberMissingConstraintExpression>());

	[Test]
	public void TypeNameCanHaveOneNumberAtEnd()
	{
		var vector2 = CreateType("Vector2", "has numbers", "AddNumbers Number",
			"\tnumbers(0) + numbers(1)");
		Assert.That(() => vector2.Name, Is.EqualTo("Vector2"));
	}

	[TestCase("2Vector")]
	[TestCase("Vector22")]
	[TestCase("Matrix0")]
	[TestCase("Matrix1")]
	public void InvalidTypeNames(string typeName) =>
		Assert.That(() => CreateType(typeName, "has numbers", "Unused Number", "\t1"),
			Throws.
				InstanceOf<
					Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers>());

	[Test]
	public void NumberInTheEndIsNotAllowedIfTypeWithoutNumberExists()
	{
		CreateType("Matrix", "has numbers", "Unused Number", "\t1");
		Assert.That(() => CreateType("Matrix2", "has numbers", "Unused Number", "\t1"),
			Throws.
				InstanceOf<
					Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers>());
	}

	[Test]
	public void AppleTypeCompatibilityCheck()
	{
		var apple = CreateType("Apple", "has name", "Quantity Number", "\tvalue.Length");
		var redApple = CreateType("RedApple", "has apple", "Color Text", "\tvalue.Color");
		Assert.That(apple.IsCompatible(redApple), Is.False);
		Assert.That(redApple.IsCompatible(apple), Is.True);
	}

	[Test]
	public void LoggerIsCompatibleWithFile()
	{
		var logger = CreateType("Logger", "has source File", "Log Number", "\tvalue");
		Assert.That(logger.IsCompatible(package.GetType(Base.File)), Is.True);
	}

	[Test]
	public void AccountantIsNotCompatibleWithFile()
	{
		var accountant = CreateType("Accountant", "has taxFile File", "has assetFile File", "Calculate Number", "\tvalue");
		Assert.That(accountant.IsCompatible(package.GetType(Base.File)), Is.False);
	}
}