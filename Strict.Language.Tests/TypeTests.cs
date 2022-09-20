using System;
using System.Collections.Generic;
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
		CreateType(Base.App, "Run");
	}

	private Type CreateType(string name, params string[] lines) =>
		new Type(package, new TypeLines(name, lines)).ParseMembersAndMethods(null!);

	private Package package = null!;

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
		Assert.That(() => CreateType("Program", " implement App"),
			Throws.InstanceOf<Type.ExtraWhitespacesFoundAtBeginningOfLine>());
		Assert.That(() => CreateType(Base.HashCode, "has\t"),
			Throws.InstanceOf<Type.ExtraWhitespacesFoundAtEndOfLine>());
	}

	[Test]
	public void TypeParsersMustStartWithImplementOrHas() =>
		Assert.That(() => CreateType(Base.Error, "Run", "\tlog.WriteLine"),
			Throws.InstanceOf<Type.TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies>());

	[Test]
	public void JustMembersAreAllowed() =>
		Assert.That(CreateType(Base.Error, "has log", "has count").Members.Count, Is.EqualTo(2));

	[Test]
	public void GetUnknownTypeWillCrash() =>
		Assert.That(() => package.GetType(Base.Computation),
			Throws.InstanceOf<Context.TypeNotFound>());

	[TestCase("implement invalidType")]
	[TestCase("has log", "Run InvalidType", "\tlet a = 5")]
	public void TypeNotFound(params string[] lines) =>
		Assert.That(() => CreateType(Base.Error, lines),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.InstanceOf<Context.TypeNotFound>());

	[Test]
	public void NoMethodsFound() =>
		Assert.That(
			() => new Type(new Package(nameof(NoMethodsFound)), new TypeLines("dummy", "has log")).
				ParseMembersAndMethods(null!), Throws.InstanceOf<Type.NoMethodsFound>());

	[Test]
	public void ExtraWhitespacesFoundAtBeginningOfLine() =>
		Assert.That(
			() => CreateType(nameof(ExtraWhitespacesFoundAtBeginningOfLine), "has log", "Run",
				" let a = 5"), Throws.InstanceOf<Type.ExtraWhitespacesFoundAtBeginningOfLine>());

	[Test]
	public void NoMatchingMethodFound() =>
		Assert.That(
			() => CreateType(nameof(NoMatchingMethodFound), "has log", "Run", "\tlet a = 5").
				GetMethod("UnknownMethod", Array.Empty<Expression>()),
			Throws.InstanceOf<Type.NoMatchingMethodFound>());

	[Test]
	public void TypeNameMustBeWord() =>
		Assert.That(() => new Member(package.GetType(Base.App), "blub7", null!),
			Throws.InstanceOf<Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers>());

	[Test]
	public void ImplementAnyIsImplicitAndNotAllowed() =>
		Assert.That(() => CreateType("Program", "implement Any"),
			Throws.InstanceOf<Type.ImplementAnyIsImplicitAndNotAllowed>());

	[Test]
	public void ImplementMustBeBeforeMembersAndMethods() =>
		Assert.That(() => CreateType("Program", "has log", "implement App"),
			Throws.InstanceOf<Type.ImplementMustComeBeforeMembersAndMethods>());

	[Test]
	public void MembersMustComeBeforeMethods() =>
		Assert.That(() => CreateType("Program", "Run", "has log"),
			Throws.InstanceOf<Type.MembersMustComeBeforeMethods>());

	[Test]
	public void SimpleApp() =>
		// @formatter:off
		CheckApp(CreateType("Program",
			"implement App",
			"has log",
			"Run",
			"\tlog.Write(\"Hello World!\")"));

	private static void CheckApp(Type program)
	{
		Assert.That(program.Implements[0].Name, Is.EqualTo(Base.App));
		Assert.That(program.Members[0].Name, Is.EqualTo("log"));
		Assert.That(program.Methods[0].Name, Is.EqualTo("Run"));
		Assert.That(program.IsTrait, Is.False);
	}

	[Test]
	public void AnotherApp() =>
		CheckApp(CreateType("Program",
			"implement App",
			"has log",
			"Run",
			"\tfor number in Range(0, 10)",
			"\t\tlog.Write(\"Counting: \" + number)"));

	[Test]
	public void MustImplementAllTraitMethods() =>
		Assert.That(() => CreateType("Program",
				"implement App",
				"add(number)",
				"\treturn one + 1"),
			Throws.InstanceOf<Type.MustImplementAllTraitMethods>());

	[Test]
	public void TraitMethodsMustBeImplemented() =>
		Assert.That(() => CreateType("Program",
				"implement App",
				"Run"),
			Throws.InstanceOf<Type.MethodMustBeImplementedInNonTraitType>());
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
	public void ImplementsWithBrackets() =>
		Assert.That(
			new TypeLines(nameof(ImplementsWithBrackets), "implement Text(Character)", "has log").
				ImplementTypes, Has.Count.EqualTo(2));

	[Test]
	public void CanUpCastNumberWithList()
	{
		var type = CreateType(nameof(CanUpCastNumberWithList), "has log",
			"Add(first Number, other List) List", "\tfirst + other");
		var result = type.FindMethod("Add",
			new List<Expression>
			{
				new Number(type, 5),
				new List(null!, new List<Expression> { new Number(type, 6), new Number(type, 7) })
			});
		Assert.That(result, Is.InstanceOf<Method>());
		Assert.That(result?.ToString(),
			Is.EqualTo("Add(first TestPackage.Number, other TestPackage.List) List"));
	}

	[TestCase("has number", "Run", "\tnumber = 1 + 1")]
	[TestCase("has number", "Run", "\tlet result = 5", "\tresult = 6")]
	public void ImmutableTypesCannotBeChanged(params string[] code) =>
		Assert.That(
			() => new Type(package, new TypeLines(nameof(ImmutableTypesCannotBeChanged), code)).ParseMembersAndMethods(new MethodExpressionParser()).Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Mutable.ImmutableTypesCannotBeChanged>());

	[TestCase("has count = 0", "Run", "\tcount = 5")]
	[TestCase("has counter = Count(0)", "Run", "\tcounter = 5")]
	public void MutableMemberTypesCanBeChanged(params string[] code)
	{
		var type = new Type(package, new TypeLines(nameof(MutableMemberTypesCanBeChanged), code)).
			ParseMembersAndMethods(new MethodExpressionParser());
		type.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(type.Members[0].Value, Is.EqualTo(new Number(type, 5)));
	}

	[TestCase("has number",
		"Run",
		"\tlet result = Count(2)",
		"\tresult = Count(5)")]
	[TestCase("has number",
		"Run",
		"\tlet result = Mutable(2)",
		"\tresult = Count(5)")]
	public void MutableVariableCanBeChanged(params string[] code)
	{
		var type = new Type(package, new TypeLines(nameof(MutableVariableCanBeChanged), code)).
			ParseMembersAndMethods(new MethodExpressionParser());
		var body = (Body)type.Methods[0].GetBodyAndParseIfNeeded();
		Assert.That(body.FindVariableValue("result")!.ToString(), Is.EqualTo("Count(5)"));
	}

	[Test]
	public void InvalidAssignmentTarget() =>
		Assert.That(
			() => new Type(package,
					new TypeLines(nameof(InvalidAssignmentTarget), "has log", "Run", "\tCount(6) = 6")).
				ParseMembersAndMethods(new MethodExpressionParser()).Methods[0].GetBodyAndParseIfNeeded(),
			Throws.InstanceOf<Mutable.InvalidAssignmentTarget>());

	/*TODO
	[Test]
	public void MakeSureGenericTypeIsProperlyGenerated()
	{
		var listType = new Type(package,
			new TypeLines(nameof(List), "has elements Generics", "Length", "\tList((1, 2)).Length is 2",
				"\telements.Length"));
		Assert.That(listType.Members[0].Type, Is.EqualTo(package.GetType(Base.List, Base.Generic)));
		var lengthBody = listType.ParseMembersAndMethods(new MethodExpressionParser()).Methods[0].
			GetBodyAndParseIfNeeded();
		Assert.That(lengthBody.ReturnType, Is.EqualTo(package.GetType(Base.Number)));
		var binary = (Binary)((Body)lengthBody).Expressions[0];
		var lengthCall = (MethodCall)binary.Instance;
		var constructorCall = lengthCall.Instance;
		Assert.That(constructorCall.ReturnType, Is.EqualTo(package.GetType(Base.List, Base.Number)));
	}
	*/
}