using NUnit.Framework;

namespace Strict.Language.Tests;

public class TypeTests
{
	[SetUp]
	public void CreatePackage()
	{
		package = new TestPackage();
		CreateType(Base.App, "Run");
	}

	private Type CreateType(string name, params string[] lines) =>
		//TODO: invent MockFileData
		new(package, new FileData(name, lines), null!);

	private Package package = null!;

	[Test]
	public void AddingTheSameNameIsNotAllowed() =>
		Assert.That(() => CreateType(Base.App, "Run"), Throws.InstanceOf<Type.TypeAlreadyExistsInPackage>());

	[Test]
	public void EmptyLineIsNotAllowed() =>
		Assert.That(() => CreateType(Base.Error, ""),
			Throws.InstanceOf<Type.EmptyLineIsNotAllowed>().With.Message.Contains("line 1"));

	[Test]
	public void WhitespacesAreNotAllowed()
	{
		Assert.That(() => CreateType(Base.Error, " "),
			Throws.InstanceOf<Type.ExtraWhitespacesFoundAtBeginningOfLine>());
		Assert.That(() => CreateType("Program", "Run", " "),
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
		Assert.That(CreateType(Base.Error, "has log", "has count").Members,
			Has.Count.EqualTo(2));

	[Test]
	public void GetUnknownTypeWillCrash() =>
		Assert.That(() => package.GetType(Base.Computation), Throws.InstanceOf<Context.TypeNotFound>());

	[Test]
	public void TypeNameMustBeWord() =>
		Assert.That(() => new Member(package.GetType(Base.App), "blub7", null!),
			Throws.InstanceOf<Context.NameMustBeAWordWithoutAnySpecialCharactersOrNumbers>());

	[Test]
	public void ImportMustBeFirst() =>
		Assert.That(() => CreateType("Program", "has number", "import TestPackage"), Throws.InstanceOf<ParsingFailed>().With.InnerException.
			InstanceOf<Type.ImportMustBeFirst>());

	[Test]
	public void ImportMustBeValidPackageName() =>
		Assert.That(() => CreateType("Program", "import $YI(*SI"),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.PackageNotFound>());

	[Test]
	public void Import()
	{
		var program = CreateType("Program", "import TestPackage", "has number", "GetNumber returns Number", "\treturn number");
		Assert.That(program.Imports[0].Name, Is.EqualTo(nameof(TestPackage)));
	}

	[Test]
	public void ImplementAnyIsImplicitAndNotAllowed() =>
		Assert.That(() => CreateType("Program", "implement Any"),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.ImplementAnyIsImplicitAndNotAllowed>());

	[Test]
	public void ImplementMustBeBeforeMembersAndMethods() =>
		Assert.That(() => CreateType("Program", "has log", "implement App"),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.ImplementMustComeBeforeMembersAndMethods>());

	[Test]
	public void MembersMustComeBeforeMethods() =>
		Assert.That(() => CreateType("Program", "Run", "has log"),
			Throws.InstanceOf<ParsingFailed>().With.InnerException.
				InstanceOf<Type.MembersMustComeBeforeMethods>());

	[Test]
	public void SimpleApp() =>
		CheckApp(CreateType("Program", "implement App", "has log", "Run",
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
		CheckApp(CreateType("Program", "implement App", "has log", "Run", "\tfor number in Range(0, 10)", "\t\tlog.Write(\"Counting: \" + number)"));

	[Test]
	public void MustImplementAllTraitMethods() =>
		Assert.That(() => CreateType("Program", "implement App", "add(number)", "\treturn one + 1"),
			Throws.InstanceOf<Type.MustImplementAllTraitMethods>());

	[Test]
	public void TraitMethodsMustBeImplemented() =>
		Assert.That(() => CreateType("Program", "implement App", "Run"),
			Throws.InstanceOf<Type.MethodMustBeImplementedInNonTraitType>());

	[Test]
	public void Trait()
	{
		var app = CreateType("DummyApp", "Run");
		Assert.That(app.IsTrait, Is.True);
		Assert.That(app.Name, Is.EqualTo("DummyApp"));
		Assert.That(app.Methods[0].Name, Is.EqualTo("Run"));
	}
}