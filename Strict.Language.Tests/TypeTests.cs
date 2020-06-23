using NUnit.Framework;

namespace Strict.Language.Tests
{
	public class TypeTests
	{
		[SetUp]
		public void CreatePackage()
		{
			package = new TestPackage();
			new Type(package, "App", "method Run");
		}

		private Package package;
		
		[Test]
		public void EmptyCodeIsNotAllowed() =>
			Assert.Throws<Type.NoCodeGiven>(() => new Type(package, Base.Count, ""));

		[Test]
		public void EmptyLineIsNotAllowed() =>
			Assert.Throws<Type.EmptyLine>(() => new Type(package, Base.Count, @"
"));

		[Test]
		public void AddingTheSameNameIsNotAllowed() =>
			Assert.Throws<Type.TypeAlreadyExistsInPackage>(() => new Type(package, "App", ""));

		[Test]
		public void WhitespacesAreNotAllowed()
		{
			Assert.Throws<Type.ExtraWhitespacesFound>(() => new Type(package, Base.Count, " "));
			Assert.Throws<Type.ExtraWhitespacesFound>(() => new Type(package, Base.HashCode, "has\t"));
		}

		[Test]
		public void LineWithOneWordIsNotAllowed() =>
			Assert.Throws<Type.LineWithJustOneWord>(() => new Type(package, Base.Count, "has"));

		[Test]
		public void TypeParsersMustStartWithImplementOrHas() =>
			Assert.Throws<Type.TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies>(() =>
				new Type(package, Base.Count, @"method Run
	log.WriteLine"));

		[Test]
		public void JustMembersIsNotValidCode() =>
			Assert.Throws<Type.NoMethodsFound>(() => new Type(package, Base.Count, @"has log
has count"));

		[Test]
		public void InvalidSyntax() =>
			Assert.Throws<Type.InvalidLine>(() => new Type(package, Base.Count, "has log\na b"));

		[Test]
		public void GetUnknownTypeWillCrash() =>
			Assert.Throws<Context.TypeNotFound>(() => package.GetType(Base.Computation));

		[Test]
		public void SimpleApp() =>
			CheckApp(new Type(package, "Program", @"implement App
has log
method Run
	log.WriteLine(""Hello World!"")"));

		private static void CheckApp(Type program)
		{
			Assert.That(program.Implements[0].Trait.Name, Is.EqualTo("App"));
			Assert.That(program.Members[0].Name, Is.EqualTo("log"));
			Assert.That(program.Methods[0].Name, Is.EqualTo("Run"));
		}

		[Test]
		public void AnotherApp() =>
			CheckApp(new Type(package, "Program", @"implement App
has log
method Run
	for number in Range(0, 10)
		log.WriteLine(""Counting: "" + number)"));

		[Test]
		public void Trait()
		{
			var app = new Type(package, "DummyApp", "method Run");
			Assert.That(app.IsTrait, Is.True);
			Assert.That(app.Name, Is.EqualTo("DummyApp"));
			Assert.That(app.Methods[0].Name, Is.EqualTo("Run"));
		}
	}
}