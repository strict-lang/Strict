using System;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using Strict.Language.Expressions;
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
		new Type(package, new TypeLines(name, lines)).ParseMembersAndMethods(null!);

	public Package package = null!;
	public ExpressionParser parser = null!;

	[Test]
	public void AddingTheSameNameIsNotAllowed() =>
		Assert.That(() => CreateType(Base.App, "Run"),
			Throws.InstanceOf<Type.TypeAlreadyExistsInPackage>());

	[Test]
	public void EmptyLineIsNotAllowed() =>
		Assert.That(() => CreateType(Base.Error, ""),
			Throws.InstanceOf<Type.EmptyLineIsNotAllowed>().With.Message.Contains("line 1"));

	[Test]
	public void WhitespacesAreNotAllowed()
	{
		Assert.That(() => CreateType(Base.Error, " "),
			Throws.InstanceOf<Type.ExtraWhitespacesFoundAtBeginningOfLine>());
		Assert.That(() => CreateType("Program", " has App"),
			Throws.InstanceOf<Type.ExtraWhitespacesFoundAtBeginningOfLine>());
		Assert.That(() => CreateType(Base.HashCode, "has\t"),
			Throws.InstanceOf<Type.ExtraWhitespacesFoundAtEndOfLine>());
	}

	[Test]
	public void TypeParsersMustStartWithMember() =>
		Assert.That(() => CreateType(Base.Error, "Run", "\tlog.WriteLine"),
			Throws.InstanceOf<Type.TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies>());

	[Test]
	public void JustMembersAreAllowed() =>
		Assert.That(CreateType(Base.Error, "has log", "mutable counter Number").Members.Count, Is.EqualTo(2));

	[Test]
	public void GetUnknownTypeWillCrash() =>
		Assert.That(() => package.GetType(Base.Computation),
			Throws.InstanceOf<Context.TypeNotFound>());

	[TestCase("has invalidType")]
	[TestCase("has log", "Run InvalidType", "\tconstant a = 5")]
	public void TypeNotFound(params string[] lines) =>
		Assert.That(() => CreateType(Base.Error, lines),
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

	/*TODO
	[TestCase("has number", "Run", "\tnumber = 1 + 1")]
	[TestCase("has number", "Run", "\tconstant result = 5", "\tresult = 6")]
	public void ImmutableTypesCannotBeChanged(params string[] code) =>
		Assert.That(
			() => new Type(package, new TypeLines(nameof(ImmutableTypesCannotBeChanged), code)).ParseMembersAndMethods(parser).Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Mutable.ImmutableTypesCannotBeChanged>());
	*/
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

	/*TODO
	[Test]
	public void InvalidAssignmentTarget() =>
		Assert.That(
			() => new Type(package,
					new TypeLines(nameof(InvalidAssignmentTarget), "has log", "Run", "\tCount(6) = 6")).
				ParseMembersAndMethods(parser).Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Mutable.InvalidAssignmentTarget>());
	*/
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
			new TypeLines("Customer", "has name Text", "has age Number", "Print Text",
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
			new TypeLines("Customer", "has name Text", "has age Number", "Print Text",
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
		Assert.That(
			range.AvailableMethods.Values.Select(methods => methods.Count).Sum(),
			Is.EqualTo(8));
	}

	[Test]
	public void TextTypeShouldHaveCorrectAvailableMethods()
	{
		var text = package.GetType(Base.Text + "s");
		Assert.That(text.AvailableMethods.Values.Select(methods => methods.Count).Sum(),
			Is.EqualTo(47));
	}

	[Test]
	public void PrivateMethodsShouldNotBeAddedToAvailableMethods()
	{
		var type = new Type(package, new TypeLines(nameof(PrivateMethodsShouldNotBeAddedToAvailableMethods),
			"has output", "run", "\tconstant n = 5"));
		type.ParseMembersAndMethods(parser);
		Assert.That(type.Methods.Count, Is.EqualTo(1));
		Console.WriteLine(type.AvailableMethods.Keys.ToWordList());
		Assert.That(type.AvailableMethods.Keys.Contains("run"), Is.False);
	}

	//TODO: support parsing of Mutable(Number)(2)
	//[Test]
	//public void CreateMutableNumberWithAnotherArgument()
	//{
	//	var type = new Type(package, new TypeLines(nameof(MutableTypesOrImplementsShouldNotBeUsedDirectlyy), "has number",
	//			"Run",
	//			"\tmutable result = Mutable(Number)(2)")).
	//		ParseMembersAndMethods(parser);
	//	Assert.That(() => type.Methods[0].GetBodyAndParseIfNeeded(),
	//		Throws.InstanceOf<Type.GenericTypesCannotBeUsedDirectlyUseImplementation>());
	//}
}