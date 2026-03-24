using List = Strict.Expressions.List;

namespace Strict.Language.Tests;

public sealed class TypeMethodFinderTests
{
	[SetUp]
	public void CreatePackage()
	{
		parser = new MethodExpressionParser();
		appType = CreateType("App", Method.Run);
	}

	private Type CreateType(string name, params string[] lines) =>
		new Type(TestPackage.Instance, new TypeLines(name, lines)).ParseMembersAndMethods(parser);

	public ExpressionParser parser = null!;
	private Type appType = null!;

	[TearDown]
	public void TearDown() => appType.Dispose();

	[Test]
	public void CanUpCastNumberWithList()
	{
		using var type = CreateType(nameof(CanUpCastNumberWithList), "has logger",
			"Add(first Number, other Numbers) List", "\tfirst + other");
		var result = type.FindMethod("Add",
		[
			new Number(type, 5),
			new List(null!, [new Number(type, 6), new Number(type, 7)])
		]);
		Assert.That(result, Is.InstanceOf<Method>());
		Assert.That(result?.ToString(),
			Is.EqualTo("Add(first TestPackage/Number, other TestPackage/List(Number)) List"));
	}

	[Test]
	public void GenericTypesCannotBeUsedDirectlyUseImplementation()
	{
		using var type = CreateType(nameof(GenericTypesCannotBeUsedDirectlyUseImplementation),
			"has generic", "AddGeneric(first Generic, other List) List", "\tfirst + other");
		var exception = Assert.Throws<Type.GenericTypesCannotBeUsedDirectlyUseImplementation>(() =>
			type.FindMethod("AddGeneric",
				[new Number(type, 6), new List(null!, [new Number(type, 7), new Number(type, 8)])]));
		Assert.That(exception!.Message,
			Does.Contain("Lookup context type:").And.Contain("Attempted method:").And.
				Contain("Arguments:").And.Contain("List(Number)"));
	}

	[Test]
	public void UsingGenericMethodIsAllowed()
	{
		using var type = CreateType(nameof(UsingGenericMethodIsAllowed), "has logger",
			"Add(other Texts, first Generic) List", "\tother + first");
		Assert.That(
			type.FindMethod("Add",
				[
					new List(null!, [new Text(type, "Hi"), new Text(type, "Hello")]), new Number(type, 5)
				])?.
				ToString(),
			Is.EqualTo(
				"Add(other TestPackage/List(Text), first TestPackage/Generic) List"));
	}

	[Test]
	public void GenericMethodShouldAcceptAllInputTypes()
	{
		using var type = CreateType(nameof(GenericMethodShouldAcceptAllInputTypes),
			"has logger",
			"Write(generic)", "\tlogger.Log(generic)");
		Assert.That(type.FindMethod("Write", [new Text(type, "hello")])?.ToString(),
			Is.EqualTo("Write(generic TestPackage/Generic)"));
		Assert.That(type.FindMethod("Write", [new Number(type, 5)])?.ToString(),
			Is.EqualTo("Write(generic TestPackage/Generic)"));
	}

	[Test]
	public void MethodParameterCanBeGeneric()
	{
		using var type = new Type(TestPackage.Instance,
			new TypeLines(nameof(MethodParameterCanBeGeneric), "has logger", "Something(input Generics)",
				"\tconstant result = input + 5")).ParseMembersAndMethods(parser);
		Assert.That(type.FindMethod("Something", [new List(null!, [new Text(type, "hello")])]),
			Is.Not.Null);
	}

	[Test]
	public void CreateTypeUsingConstructorMembers()
	{
		using var _ = new Type(TestPackage.Instance,
			new TypeLines("Customer", "has text", "has age Number", "Print Text",
				"\t\"Customer Name: \" + name + \" Age: \" + age")).ParseMembersAndMethods(parser);
		using var createCustomer = new Type(TestPackage.Instance,
			new TypeLines(nameof(CreateTypeUsingConstructorMembers), "has logger", "Something",
				"\tconstant customer = Customer(\"Ben\", 28)", "\tcustomer is Customer")).ParseMembersAndMethods(parser);
		var body = (Body)createCustomer.Methods[0].GetBodyAndParseIfNeeded();
		var assignment = (Declaration)body.Expressions[0];
		Assert.That(assignment.Value.ReturnType.Name, Is.EqualTo("Customer"));
		Assert.That(assignment.Value.ToString(), Is.EqualTo("Customer(\"Ben\", 28)"));
	}

	[Test]
	public void UsingToMethodForComplexTypeConstructorIsForbidden()
	{
		using var _ = new Type(TestPackage.Instance,
			new TypeLines("Customer", "has text", "has age Number", "Print Text",
				"\t\"Customer Name: \" + name + \" Age: \" + age")).ParseMembersAndMethods(parser);
		using var createCustomer = new Type(TestPackage.Instance,
			new TypeLines(nameof(CreateTypeUsingConstructorMembers), "has logger", "Something",
				"\tconstant customer = (\"Ben\", 28) to Customer")).ParseMembersAndMethods(parser);
		// ReSharper disable once AccessToDisposedClosure
		Assert.That(() => createCustomer.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<List.ListElementsMustHaveMatchingType>());
	}

	[Test]
	public void CreateStacktraceTypeUsingMembersInConstructor()
	{
		using var logger = new Type(TestPackage.Instance,
			new TypeLines("MethodLogger",
				"has logger",
				"has method",
				"Log",
				"\tlogger.Log(stacktrace to Text)",
				"GetStacktrace Stacktrace",
				"\tStacktrace(method, \"filePath\", 5)")).ParseMembersAndMethods(parser);
		var stackTraceMethodReturnType = logger.Methods[1].ReturnType;
		Assert.That(stackTraceMethodReturnType.Name, Is.EqualTo("Stacktrace"));
		Assert.That(stackTraceMethodReturnType.Members.Count, Is.EqualTo(3));
	}

	[Test]
	public void MutableTypesOrImplementsShouldNotBeUsedDirectly()
	{
		using var type = new Type(TestPackage.Instance,
			new TypeLines(nameof(MutableTypesOrImplementsShouldNotBeUsedDirectly), "has number",
				"Run", "\tmutable result = Mutable(2)")).ParseMembersAndMethods(parser);
		Assert.That(() => type.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.GenericTypesCannotBeUsedDirectlyUseImplementation>());
	}

	[Test]
	public void RangeTypeShouldHaveCorrectAvailableMethods()
	{
		var range = TestPackage.Instance.GetType(Type.Range);
		Assert.That(range.AvailableMethods.Values.Select(methods => methods.Count).Sum(),
			Is.EqualTo(8),
			"AvailableMethods: " + range.AvailableMethods.DictionaryToWordList("\n"));
	}

	[Test]
	public void TextTypeShouldHaveCorrectAvailableMethods()
	{
		var text = TestPackage.Instance.GetType(Type.Text + "s");
		Assert.That(text.AvailableMethods.Values.Select(methods => methods.Count).Sum(),
			Is.GreaterThanOrEqualTo(18),
			"AvailableMethods: " + text.AvailableMethods.DictionaryToWordList("\n"));
	}

	[Test]
	public void DictionaryIsComparisonShouldNotThrow()
	{
		var number = TestPackage.Instance.GetType(Type.Number);
		var dictionary = TestPackage.Instance.GetType(Type.Dictionary).
			GetGenericImplementation(number, number);
		Assert.That(dictionary.FindMethod(BinaryOperator.Is, [new Instance(dictionary)]),
			Is.Not.Null);
	}

	[Test]
	public void PrivateMethodsShouldNotBeAddedToAvailableMethods()
	{
		using var type = new Type(TestPackage.Instance, new TypeLines(nameof(PrivateMethodsShouldNotBeAddedToAvailableMethods),
			"has textWriter", "run", "\tconstant n = 5"));
		type.ParseMembersAndMethods(parser);
		Assert.That(type.Methods.Count, Is.EqualTo(1));
		Assert.That(type.AvailableMethods.Keys.Contains("run"), Is.False);
	}

	[Test]
	public void AvailableMethodsShouldNotHaveMembersPrivateMethods()
	{
		using var _ = new Type(TestPackage.Instance,
			new TypeLines("ProgramWithPublicAndPrivateMethods", "has logger", "PublicMethod", "\tlogger.Log(\"I am exposed\")", "privateMethod", "\tlogger.Log(\"Support privacy\")")).ParseMembersAndMethods(parser);
		using var type = new Type(TestPackage.Instance,
			new TypeLines(nameof(AvailableMethodsShouldNotHaveMembersPrivateMethods),
				"has programWithPublicAndPrivateMethods", "run", "\tconstant n = 5"));
		type.ParseMembersAndMethods(parser);
		Assert.That(type.AvailableMethods.Keys.Contains("privateMethod"), Is.False);
		Assert.That(type.AvailableMethods.Keys.Contains("PublicMethod"), Is.True);
		Assert.That(
			type.AvailableMethods.Values.Any(methods =>
				methods.Any(method => !method.IsPublic && !method.Name.AsSpan().IsOperator())), Is.False,
			// If this fails, check by debugging each private method and see if IsOperator returns true
			type.AvailableMethods.DictionaryToWordList("\n"));
	}

	[Test]
	public void IsMutableAndHasMatchingInnerType()
	{
		var number = TestPackage.Instance.GetType(Type.Number);
		Assert.That(
			TestPackage.Instance.GetType(Type.Mutable).GetGenericImplementation(number).
				IsSameOrCanBeUsedAs(number), Is.True);
		var type = TestPackage.Instance.GetType(Type.Text);
		Assert.That(
			TestPackage.Instance.GetType(Type.Mutable).GetGenericImplementation(type).
				IsSameOrCanBeUsedAs(number), Is.False);
	}

	[Test]
	public void InitializeInnerTypeMemberUsingOuterTypeConstructor()
	{
		using var thing = CreateType("Thing", "has character", "SomeThing Number", "\tvalue");
		using var superThing = CreateType("SuperThing", "has thing", "SuperSomeThing Number", "\tvalue");
		using var superThingUser = CreateType("SuperThingUser", "has superThing = SuperThing(7)",
			"UseSuperThing Number",
			"\tsuperThing to Number is \"7\" to Number",
			"\tsuperThing is 7",
			"\tsuperThing");
		superThingUser.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(superThingUser.Members[0].Type, Is.EqualTo(TestPackage.Instance.GetType("SuperThing")));
	}

	[Test]
	public void GetGenericImplementationCanUseCustomFromMethod()
	{
		using var _ = CreateType("Comparer", "has FirstTypes Generics", "has SecondType Generic",
			"from(number)", "\tComparer(number, number)", "Compare", "\tfirstType is secondType");
		using var comparerImplementation = new Type(TestPackage.Instance,
			new TypeLines(nameof(GetGenericImplementationCanUseCustomFromMethod),
				"has custom Comparer(Number)", "UnusedMethod Number", "\t5"));
		comparerImplementation.ParseMembersAndMethods(new MethodExpressionParser());
	}

	[Test]
	public void NumberCanBePassedInAsText()
	{
		using var type = CreateType(nameof(NumberCanBePassedInAsText), "has logger",
			"Run", "\tlogger.Log(5)");
		var method = type.FindMethod("Run", []);
		Assert.That(method, Is.Not.Null);
		var call = (MethodCall)method!.GetBodyAndParseIfNeeded();
		Assert.That(call.ToString(), Is.EqualTo("logger.Log(5)"));
	}

	[Test]
	public void SingleCharacterTextIsAlwaysValidAsCharacter()
	{
		using var type = CreateType(nameof(SingleCharacterTextIsAlwaysValidAsCharacter), "has logger",
			"Run", "\t5 to Character is \"5\"");
		type.GetMethod("Run", []).GetBodyAndParseIfNeeded();
	}

	[Test]
	public void ConstraintWithLengthGreaterThanZero()
	{
		using var type = CreateType(nameof(ConstraintWithLengthGreaterThanZero),
			"has logger",
			"Result(items Generics) Text",
			"\titems.Length > 0 then \"Has items\" else \"No items\"");
		var method = type.FindMethod("Result", [new List(null!, [new Number(type, 1)])]);
		Assert.That(method, Is.Not.Null);
	}

	[Test]
	public void ConstraintWithExactLength()
	{
		using var type = CreateType(nameof(ConstraintWithExactLength),
			"has logger",
			"Pair(items Generics) Text",
			"\tvalue.Length is 2 then \"Pair\" else \"Not pair\"");
		var method = type.FindMethod("Pair", [new List(null!, [new Number(type, 1), new Number(type, 2)])]);
		Assert.That(method, Is.Not.Null);
	}
}