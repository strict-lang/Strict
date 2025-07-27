using NUnit.Framework;
using Strict.Expressions;
using List = Strict.Expressions.List;

namespace Strict.Language.Tests;

public sealed class TypeMethodFinderTests
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
	public void CanUpCastNumberWithList()
	{
		var type = CreateType(nameof(CanUpCastNumberWithList), "has logger",
			"Add(first Number, other Numbers) List", "\tfirst + other");
		var result = type.FindMethod("Add",
		[
			new Number(type, 5),
			new List(null!, [new Number(type, 6), new Number(type, 7)])
		]);
		Assert.That(result, Is.InstanceOf<Method>());
		Assert.That(result?.ToString(),
			Is.EqualTo("Add(first TestPackage.Number, other TestPackage.List(TestPackage.Number)) List"));
	}

	[Test]
	public void GenericTypesCannotBeUsedDirectlyUseImplementation()
	{
		var type = CreateType(nameof(GenericTypesCannotBeUsedDirectlyUseImplementation),
			"has generic", "AddGeneric(first Generic, other List) List", "\tfirst + other");
		Assert.That(
			() => type.FindMethod("AddGeneric",
				[new Number(type, 6), new List(null!, [new Number(type, 7), new Number(type, 8)])]),
			Throws.InstanceOf<Type.GenericTypesCannotBeUsedDirectlyUseImplementation>());
	}

	[Test]
	public void UsingGenericMethodIsAllowed()
	{
		var type = CreateType(nameof(CanUpCastNumberWithList), "has logger",
			"Add(other Texts, first Generic) List", "\tother + first");
		Assert.That(
			type.FindMethod("Add",
				[
					new List(null!, [new Text(type, "Hi"), new Text(type, "Hello")]), new Number(type, 5)
				])?.
				ToString(),
			Is.EqualTo(
				"Add(other TestPackage.List(TestPackage.Text), first TestPackage.Generic) List"));
	}

	[Test]
	public void GenericMethodShouldAcceptAllInputTypes()
	{
		var type = CreateType(nameof(GenericMethodShouldAcceptAllInputTypes),
			"has logger",
			"Write(generic)", "\tlogger.Log(generic)");
		Assert.That(type.FindMethod("Write", [new Text(type, "hello")])?.ToString(),
			Is.EqualTo("Write(generic TestPackage.Generic)"));
		Assert.That(type.FindMethod("Write", [new Number(type, 5)])?.ToString(),
			Is.EqualTo("Write(generic TestPackage.Generic)"));
	}

	[Test]
	public void MethodParameterCanBeGeneric()
	{
		var type = new Type(package,
			new TypeLines(nameof(MethodParameterCanBeGeneric), "has logger", "Something(input Generics)",
				"\tconstant result = input + 5")).ParseMembersAndMethods(parser);
		Assert.That(type.FindMethod("Something", [new List(null!, [new Text(type, "hello")])]),
			Is.Not.Null);
	}

	[Test]
	public void CreateTypeUsingConstructorMembers()
	{
		new Type(package,
			new TypeLines("Customer", "has text", "has age Number", "Print Text",
				"\t\"Customer Name: \" + name + \" Age: \" + age")).ParseMembersAndMethods(parser);
		var createCustomer = new Type(package,
			new TypeLines(nameof(CreateTypeUsingConstructorMembers), "has logger", "Something",
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
			new TypeLines(nameof(CreateTypeUsingConstructorMembers), "has logger", "Something",
				"\tconstant customer = (\"Murali\", 28) to Customer")).ParseMembersAndMethods(parser);
		Assert.That(() => createCustomer.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<List.ListElementsMustHaveMatchingType>());
	}

	[Test]
	public void CreateStacktraceTypeUsingMembersInConstructor()
	{
		var logger = new Type(package,
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
		var type =
			new Type(package,
				new TypeLines(nameof(MutableTypesOrImplementsShouldNotBeUsedDirectly), "has number",
					"Run", "\tmutable result = Mutable(2)")).ParseMembersAndMethods(parser);
		Assert.That(() => type.Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.GenericTypesCannotBeUsedDirectlyUseImplementation>());
	}

	[Test]
	public void RangeTypeShouldHaveCorrectAvailableMethods()
	{
		var range = package.GetType(Base.Range);
		Assert.That(range.AvailableMethods.Values.Select(methods => methods.Count).Sum(),
			Is.EqualTo(9), "AvailableMethods: " + range.AvailableMethods.ToWordList());
	}

	[Test]
	public void TextTypeShouldHaveCorrectAvailableMethods()
	{
		var text = package.GetType(Base.Text + "s");
		Assert.That(text.AvailableMethods.Values.Select(methods => methods.Count).Sum(),
			Is.GreaterThanOrEqualTo(18), "AvailableMethods: " + text.AvailableMethods.ToWordList());
	}

	[Test]
	public void PrivateMethodsShouldNotBeAddedToAvailableMethods()
	{
		var type = new Type(package, new TypeLines(nameof(PrivateMethodsShouldNotBeAddedToAvailableMethods),
			"has textWriter", "run", "\tconstant n = 5"));
		type.ParseMembersAndMethods(parser);
		Assert.That(type.Methods.Count, Is.EqualTo(1));
		Assert.That(type.AvailableMethods.Keys.Contains("run"), Is.False);
	}

	[Test]
	public void AvailableMethodsShouldNotHaveMembersPrivateMethods()
	{
		new Type(package,
			new TypeLines("ProgramWithPublicAndPrivateMethods", "has logger", "PublicMethod", "\tlogger.Log(\"I am exposed\")", "privateMethod", "\tlogger.Log(\"Support privacy\")")).ParseMembersAndMethods(parser);
		var type = new Type(package,
			new TypeLines(nameof(AvailableMethodsShouldNotHaveMembersPrivateMethods),
				"has programWithPublicAndPrivateMethods", "run", "\tconstant n = 5"));
		type.ParseMembersAndMethods(parser);
		Assert.That(type.AvailableMethods.Keys.Contains("privateMethod"), Is.False);
		Assert.That(type.AvailableMethods.Keys.Contains("PublicMethod"), Is.True);
		Assert.That(
			type.AvailableMethods.Values.Any(methods =>
				methods.Any(method => !method.IsPublic && !method.Name.AsSpan().IsOperator())), Is.False,
			// If this fails, check by debugging each private method and see if IsOperator returns true
			type.AvailableMethods.ToWordList());
	}

	[Test]
	public void IsMutableAndHasMatchingInnerType()
	{
		var number = package.GetType(Base.Number);
		Assert.That(CreateMutableType(Base.Number).IsSameOrCanBeUsedAs(number), Is.True);
		Assert.That(CreateMutableType(Base.Text).IsSameOrCanBeUsedAs(number), Is.False);
	}

	private Type CreateMutableType(string typeName) =>
		package.GetType(Base.Mutable).GetGenericImplementation(package.GetType(typeName));

	[Test]
	public void InitializeInnerTypeMemberUsingOuterTypeConstructor()
	{
		CreateType("Thing", "has character", "SomeThing Number", "\tvalue");
		CreateType("SuperThing", "has thing", "SuperSomeThing Number", "\tvalue");
		var superThingUser = CreateType("SuperThingUser", "has superThing = SuperThing(7)",
			"UseSuperThing Number",
			"\tsuperThing to Number is \"7\" to Number",
			"\tsuperThing is 7",
			"\tsuperThing");
		superThingUser.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(superThingUser.Members[0].Type, Is.EqualTo(package.GetType("SuperThing")));
	}

	[Test]
	public void GetGenericImplementationCanUseCustomFromMethod()
	{
		CreateType("Comparer", "has FirstTypes Generics", "has SecondType Generic",
			"from(number)", "\tComparer(number, number)", "Compare", "\tfirstType is secondType");
		var comparerImplementation = new Type(package,
			new TypeLines(nameof(GetGenericImplementationCanUseCustomFromMethod),
				"has custom Comparer(Number)", "UnusedMethod Number", "\t5"));
		comparerImplementation.ParseMembersAndMethods(new MethodExpressionParser());
	}

	[Test]
	public void NumberCanBePassedInAsText()
	{
		var type = CreateType(nameof(NumberCanBePassedInAsText), "has logger",
			"Run", "\tlogger.Log(5)");
		var method = type.FindMethod("Run", []);
		Assert.That(method, Is.Not.Null);
		var call = (MethodCall)method!.GetBodyAndParseIfNeeded();
		Assert.That(call.ToString(), Is.EqualTo("logger.Log(5)"));
	}
}