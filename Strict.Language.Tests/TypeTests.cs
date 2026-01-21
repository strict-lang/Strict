using List = Strict.Expressions.List;

namespace Strict.Language.Tests;

public sealed class TypeTests
{
	[SetUp]
	public void CreateParser()
	{
		parser = new MethodExpressionParser();
		appType = CreateType(Base.App, "Run");
	}

	private Type CreateType(string name, params string[] lines) =>
		new Type(package, new TypeLines(name, lines)).ParseMembersAndMethods(parser);

	private readonly Package package = TestPackage.Instance;
	public ExpressionParser parser = null!;
	private Type appType = null!;

	[TearDown]
	public void TearDown() => appType.Dispose();

	[Test]
	public void AddingTheSameNameIsNotAllowed() =>
		Assert.That(() => CreateType(Base.App, "Run"),
			Throws.InstanceOf<Type.TypeAlreadyExistsInPackage>());

	[Test]
	public void TypeMustStartWithMember() =>
		Assert.That(() => CreateType(nameof(TypeMustStartWithMember), "Run", "\tlogger.Log"),
			Throws.InstanceOf<Type.TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies>());

	[Test]
	public void JustMembersAreAllowed()
	{
		using var type = CreateType(nameof(JustMembersAreAllowed), "has logger",
			"mutable counter Number");
		Assert.That(type.Members.Count, Is.EqualTo(2));
	}

	[Test]
	public void GetUnknownTypeWillCrash() =>
		Assert.That(() => TestPackage.Instance.GetType(UnknownComputation),
			Throws.InstanceOf<Context.TypeNotFound>());

	private const string UnknownComputation = nameof(UnknownComputation);

	[TestCase("has invalidType")]
	[TestCase("has logger", "Run InvalidType", "\tconstant a = 5")]
	public void TypeNotFound(params string[] lines) =>
		Assert.That(() =>
			{
				using var _ = CreateType(nameof(TypeNotFound) + lines[0][5], lines);
			}, //ncrunch: no coverage
			Throws.InstanceOf<ParsingFailed>().With.InnerException.InstanceOf<Context.TypeNotFound>());

	[Test]
	public void NoMethodsFound() =>
		Assert.That(
			() => new Type(new Package(nameof(NoMethodsFound)), new TypeLines("dummy", "has Number")).
				ParseMembersAndMethods(null!), Throws.InstanceOf<Type.NoMethodsFound>());

	[Test]
	public void NoMatchingMethodFound() =>
		Assert.That(
			() => CreateType(nameof(NoMatchingMethodFound), "has logger", "Run", "\tconstant a = 5").
				GetMethod("UnknownMethod", []),
			Throws.InstanceOf<Type.NoMatchingMethodFound>());

	[Test]
	public void TypeNameMustBeWord() =>
		Assert.That(() => new Member(package.GetType(Base.App), "blub7", null!),
			Throws.InstanceOf<Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers>());

	[Test]
	public void TraitMethodsMustBeImplemented() =>
		Assert.That(() => CreateType(nameof(TraitMethodsMustBeImplemented),
				"has App",
				"Run"),
			Throws.InstanceOf<TypeParser.MethodMustBeImplementedInNonTrait>());

	[TestCase("has logger", "Run", "\tconstant result = Any")]
	[TestCase("has logger", "Run", "\tconstant result = Any(5)")]
	[TestCase("has logger", "Run", "\tconstant result = 5 + Any(5)")]
	public void VariableWithTypeAnyIsNotAllowed(params string[] lines)
	{
		using var type = new Type(package, new TypeLines(nameof(VariableWithTypeAnyIsNotAllowed), lines)).ParseMembersAndMethods(parser);
		// ReSharper disable once AccessToDisposedClosure
		Assert.That(() => type.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<MethodExpressionParser.ExpressionWithTypeAnyIsNotAllowed>().With.Message.
				Contains("Any"));
	}

	[TestCase("has logger", "Run(any)", "\tconstant result = 5")]
	[TestCase("has logger", "Run(input Any)", "\tconstant result = 5")]
	public void MethodParameterWithTypeAnyIsNotAllowed(params string[] lines) =>
		Assert.That(() =>
			{
				using var _ = CreateType(nameof(MethodParameterWithTypeAnyIsNotAllowed), lines);
			}, //ncrunch: no coverage
			Throws.InstanceOf<Method.ParametersWithTypeAnyIsNotAllowed>());

	[Test]
	public void MethodReturnTypeAsAnyIsNotAllowed() =>
		Assert.That(
			() => CreateType(nameof(MethodReturnTypeAsAnyIsNotAllowed), "has logger", "Run Any",
				"\tconstant result = 5"), Throws.InstanceOf<Method.MethodReturnTypeAsAnyIsNotAllowed>());

	[Test]
	public void SimpleApp() =>
		CheckApp(CreateType(nameof(SimpleApp),
			"has App",
			"has logger",
			"Run",
			"\tlogger.Log(\"Hello World!\")"));

	private static void CheckApp(Type program)
	{
		Assert.That(program.Members[0].Type.Name, Is.EqualTo(Base.App));
		Assert.That(program.Members[1].Name, Is.EqualTo("logger"));
		Assert.That(program.Methods[0].Name, Is.EqualTo("Run"));
		Assert.That(program.IsTrait, Is.False);
	}

	[Test]
	public void AnotherApp() =>
		CheckApp(CreateType(nameof(AnotherApp),
			"has App",
			"has logger",
			"Run",
			"\tfor number in Range(0, 10)",
			"\t\tlogger.Log(\"Counting: \" + number)"));

	[Test]
	public void NotImplementingAnyTraitMethodsAreAllowed() =>
		Assert.That(() => CreateType(nameof(NotImplementingAnyTraitMethodsAreAllowed),
			"has App",
			"add(number)",
			"\tone + 1"), Is.Not.Null);

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
	public void Trait()
	{
		var app = CreateType(nameof(Trait) + "DummyApp", "Run");
		Assert.That(app.IsTrait, Is.True);
		Assert.That(app.Name, Is.EqualTo(nameof(Trait) + "DummyApp"));
		Assert.That(app.Methods[0].Name, Is.EqualTo("Run"));
	}

	[TestCase(Base.Number, "has number", "Run", "\tmutable result = 2", "\tresult = result + 2")]
	[TestCase(Base.Text, "has number", "Run", "\tmutable result = \"2\"", "\tresult = result + \"!\"")]
	public void MutableTypesHaveProperDataReturnType(string expected, params string[] code)
	{
		using var type = new Type(package, new TypeLines(nameof(MutableTypesHaveProperDataReturnType), code));
		var body = (Body)type.ParseMembersAndMethods(parser).Methods[0].GetBodyAndParseIfNeeded();
		var expression = (Declaration)body.Expressions[0];
		Assert.That(expression.Value.ReturnType.Name, Is.EqualTo(expected));
	}

	[TestCase("has inputValue = 5", "Run", "\tinputValue = 1 + 1")]
	[TestCase("has number", "Run", "\tconstant result = 5", "\tresult = 6")]
	public void ImmutableTypesCannotBeChanged(params string[] code) =>
		Assert.That(
			() =>
			{
				using var type = new Type(package, new TypeLines(nameof(ImmutableTypesCannotBeChanged), code));
				return type.ParseMembersAndMethods(parser).Methods[0].GetBodyAndParseIfNeeded();
			}, //ncrunch: no coverage
			Throws.InstanceOf<Body.ValueIsNotMutableAndCannotBeChanged>());

	[TestCase("mutable canBeModified = 0", "Run", "\tcanBeModified = 5")]
	[TestCase("mutable counter = 0", "Run", "\tcounter = 5")]
	public void MutableMemberTypesCanBeChanged(params string[] code)
	{
		using var type = new Type(package, new TypeLines(nameof(MutableMemberTypesCanBeChanged), code)).
			ParseMembersAndMethods(parser);
		type.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(type.Members[0].InitialValue, Is.EqualTo(new Number(type, 0)));
	}

	[Test]
	public void MutableVariableCanBeChangedButNotChangeAtParseTime()
	{
		using var type = new Type(package, new TypeLines(nameof(MutableVariableCanBeChangedButNotChangeAtParseTime), "has number",
				"Run",
				"\tmutable result = 2",
				"\tresult = 5")).
			ParseMembersAndMethods(parser);
		var body = (Body)type.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.FindVariable("result")!.InitialValue.ToString(), Is.EqualTo("2"));
	}

	[Test]
	public void ValueTypeNotMatchingWithAssignmentType() =>
		Assert.That(
			() => new Type(package,
					new TypeLines(nameof(ValueTypeNotMatchingWithAssignmentType), "has logger", "Run",
						"\tlogger.Log(\"Hi\") = 6")).ParseMembersAndMethods(parser).Methods[0].
				GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<MutableReassignment.ValueTypeNotMatchingWithAssignmentType>());

	[Test]
	public void MakeSureGenericTypeIsProperlyGenerated()
	{
		var listType = package.GetType(Base.List);
		Assert.That(listType.IsGeneric, Is.True);
		Assert.That(listType.Members[0].Type, Is.EqualTo(package.GetType(Base.Iterator)));
		using var type = new Type(package,
			new TypeLines(nameof(MakeSureGenericTypeIsProperlyGenerated), "has numbers",
				"GetNumbers Numbers", "\tnumbers"));
		var getNumbersBody = type.ParseMembersAndMethods(parser).Methods[0].
			GetBodyAndParseIfNeeded();
		var numbersType = package.GetListImplementationType(package.GetType(Base.Number));
		Assert.That(getNumbersBody.ReturnType, Is.EqualTo(numbersType));
		Assert.That(numbersType.Generic, Is.EqualTo(package.GetType(Base.List)));
		Assert.That(numbersType.ImplementationTypes[0], Is.EqualTo(package.GetType(Base.Number)));
	}

	[Test]
	public void CannotGetGenericImplementationOnNonGenericType() =>
		Assert.That(
			() => package.GetType(Base.Text).GetGenericImplementation(package.GetType(Base.Number)),
			Throws.InstanceOf<Type.CannotGetGenericImplementationOnNonGeneric>());

	[Test]
	public void NonGenericExpressionCannotBeGeneric() =>
		Assert.That(
			() => new Type(package,
					new TypeLines(nameof(NonGenericExpressionCannotBeGeneric), "has list", "Something",
						"\tconstant result = list + 5")).ParseMembersAndMethods(parser).Methods[0].
				GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.GenericTypesCannotBeUsedDirectlyUseImplementation>());

	[Test]
	public void InvalidProgram() =>
		Assert.That(
			() => new Type(package,
				new TypeLines(nameof(InvalidProgram), "has list", "Something41",
					"\tconstant result = list + 5")).ParseMembersAndMethods(null!),
			Throws.InstanceOf<ParsingFailed>());

	[TestCase(Base.TextWriter, 0)]
	[TestCase(Base.Mutable, 1)]
	[TestCase(Base.Logger, 1)]
	[TestCase(Base.Number, 0)]
	[TestCase(Base.Character, 2)]
	[TestCase(Base.Text, 2)]
	[TestCase(Base.Error, 4)]
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
		using var memberWithConstraintType = CreateType(nameof(MemberWithConstraintsUsingWithKeyword),
			code, "AddNumbers Number", "\tnumbers(0) + numbers(1)");
		var member = memberWithConstraintType.Members[0];
		Assert.That(member.Type.Name, Is.EqualTo("List(TestPackage.Number)"));
		Assert.That(member.Constraints?.Length, Is.EqualTo(1));
		Assert.That(member.Constraints?[0].ToString(), Is.EqualTo("Length is 2"));
	}

	[Test]
	public void MutableMemberWithConstraintsUsingWithKeyword()
	{
		using var memberWithConstraintType =
			CreateType(nameof(MutableMemberWithConstraintsUsingWithKeyword),
				"mutable something with Length is 2 = (1, 2)", "AddNumbers Number",
				"\tnumbers(0) + numbers(1)");
		var member = memberWithConstraintType.Members[0];
		Assert.That(member.Name, Is.EqualTo("something"));
		Assert.That(member.Type.Name, Is.EqualTo("List(TestPackage.Number)"));
		Assert.That(member.Constraints?.Length, Is.EqualTo(1));
		Assert.That(member.Constraints?[0].ToString(), Is.EqualTo("Length is 2"));
		Assert.That(member.InitialValue, Is.InstanceOf<List>());
		Assert.That(member.InitialValue?.ToString(), Is.EqualTo("(1, 2)"));
	}

	[Test]
	public void MemberWithMultipleConstraintsUsingAndKeyword()
	{
		using var memberWithConstraintType =
			CreateType(nameof(MemberWithMultipleConstraintsUsingAndKeyword),
				"mutable numbers with Length is 2 and value(0) > 0", "AddNumbers Number",
				"\tnumbers(0) + numbers(1)");
		var member = memberWithConstraintType.Members[0];
		Assert.That(member.Name, Is.EqualTo("numbers"));
		Assert.That(member.Type.Name, Is.EqualTo("List(TestPackage.Number)"));
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
	public void TypeNameCanHaveOneNumberAtEnd()
	{
		using var vector2 = CreateType("Vector2", "has numbers", "AddNumbers Number",
			"\tnumbers(0) + numbers(1)");
		// ReSharper disable once AccessToDisposedClosure
		Assert.That(() => vector2.Name, Is.EqualTo("Vector2"));
	}

	/// <summary>
	/// Types are not allowed to start with numbers or non-letter characters. If they end with a
	/// number, there must be no other type that already occupies that name without a number.
	/// </summary>
	[TestCase("2Vector")]
	[TestCase("Vector22")]
	[TestCase("Matrix0")]
	[TestCase("Matrix1")]
	public void InvalidTypeNames(string typeName) =>
		Assert.That(() => CreateType(typeName, "has numbers", "Unused Number", "\t1"),
			Throws.InstanceOf<Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers>());

	[Test]
	public void NumberInTheEndIsNotAllowedIfTypeWithoutNumberExists()
	{
		using var _ = CreateType("Matrix", "has numbers", "Unused Number", "\t1");
		Assert.That(() => CreateType("Matrix2", "has numbers", "Unused Number", "\t1"),
			Throws.InstanceOf<Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers>());
	}

	[Test]
	public void AppleTypeCompatibilityCheck()
	{
		using var apple = CreateType("Apple", "has name", "Quantity Number", "\tvalue.Length");
		using var redApple = CreateType("RedApple", "has apple", "Color Text", "\tvalue.Color");
		Assert.That(apple.IsSameOrCanBeUsedAs(redApple), Is.False);
		Assert.That(redApple.IsSameOrCanBeUsedAs(apple), Is.True);
		Assert.That(redApple.IsSameOrCanBeUsedAs(package.GetType(Base.Text)), Is.True);
		Assert.That(redApple.IsSameOrCanBeUsedAs(package.GetType(Base.Number)), Is.False);
	}

	[Test]
	public void FileLoggerIsCompatibleWithFileAndLogger()
	{
		using var logger = CreateType("FileLogger", "has source File", "has logger", "Log Number",
			"\tvalue");
		Assert.That(logger.IsSameOrCanBeUsedAs(package.GetType(Base.File)), Is.True);
		Assert.That(logger.IsSameOrCanBeUsedAs(package.GetType(Base.Logger)), Is.True);
	}

	[Test]
	public void AccountantIsNotCompatibleWithFile()
	{
		using var accountant = CreateType("Accountant", "has taxFile File", "has assetFile File", "Calculate Number", "\tvalue");
		Assert.That(accountant.IsSameOrCanBeUsedAs(package.GetType(Base.File)), Is.False);
	}

	[Test]
	public void EnumCanBeUsedAsNumber()
	{
		using var instructionType = new Type(package,
			new TypeLines("Instruction", "constant Set", "constant Add")).ParseMembersAndMethods(parser);
		Assert.That(instructionType.IsSameOrCanBeUsedAs(package.GetType(Base.Number)), Is.True);
	}

	[Test]
	public void MemberNameAsAnotherMemberTypeNameIsForbidden() =>
		Assert.That(
			() => CreateType(nameof(MemberNameAsAnotherMemberTypeNameIsForbidden), "has Range",
				"has input = Range(5, 10)", "Unused", "\t1"),
			Throws.InstanceOf<MethodExpressionParser.CannotAccessMemberBeforeTypeIsParsed>());

	[TestCase(Base.Number, false)]
	[TestCase(Base.Number + "s", true)]
	[TestCase(Base.Character, false)]
	[TestCase(Base.Character + "s", true)]
	[TestCase(Base.Text, true)]
	[TestCase(Base.Text + "s", true)]
	[TestCase(Base.Boolean, false)]
	public void ValidateIsIterator(string name, bool expected) =>
		Assert.That(package.GetType(name).IsIterator, Is.EqualTo(expected));

	[Test]
	public void FindLineNumber() =>
		Assert.That(package.GetType(Base.Number).FindLineNumber("is in(other) Boolean"),
			Is.EqualTo(16));

	//ncrunch: no coverage start
	[Test]
	[Category("Slow")]
	public void CheckGetTypeCache()
	{
		using var cachedType = CreateType("CachedType", "Run");
		for (var count = 0; count < 5; count++)
			Assert.That(package.GetType("CachedType"), Is.EqualTo(cachedType));
	}
}