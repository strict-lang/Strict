using NUnit.Framework;

namespace Strict.Language.Tests
{
	public class TypeTests
	{
		[SetUp]
		public void CreatePackage()
		{
			package = new TestPackage();
			new Type(package, Base.App, "method Run");
		}

		private Package package;
		
		[Test]
		public void EmptyCodeIsNotAllowed() =>
			Assert.Throws<Type.NoCodeGiven>(() => new Type(package, Base.Count, ""));

		[Test]
		public void AddingTheSameNameIsNotAllowed() =>
			Assert.Throws<Type.TypeAlreadyExistsInPackage>(() => new Type(package, "App", ""));

		[Test]
		public void EmptyLineIsNotAllowed() =>
			Assert.That(
				Assert.Throws<Type.ParsingFailed>(() => new Type(package, Base.Count, "\n")).
					InnerException, Is.TypeOf<Type.EmptyLine>());

		[Test]
		public void WhitespacesAreNotAllowed()
		{
			Assert.That(
				Assert.Throws<Type.ParsingFailed>(() => new Type(package, Base.Count, " ")).
					InnerException, Is.TypeOf<Type.ExtraWhitespacesFound>());
			Assert.That(
				Assert.Throws<Type.ParsingFailed>(() =>
					new Type(package, Base.HashCode, "has\t")).InnerException,
				Is.TypeOf<Type.ExtraWhitespacesFound>());
		}

		[Test]
		public void LineWithOneWordIsNotAllowed() =>
			Assert.That(
				Assert.Throws<Type.ParsingFailed>(() => new Type(package, Base.Count, "has")).
					InnerException, Is.TypeOf<Type.LineWithJustOneWord>());

		[Test]
		public void TypeParsersMustStartWithImplementOrHas() =>
			Assert.That(Assert.Throws<Type.ParsingFailed>(() => new Type(package, Base.Count,
					@"method Run
	log.WriteLine")).InnerException,
				Is.TypeOf<Type.TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies>());

		[Test]
		public void JustMembersIsNotValidCode() =>
			Assert.That(Assert.Throws<Type.ParsingFailed>(() => new Type(package, Base.Count,
				@"has log
has count")).InnerException, Is.TypeOf<Type.NoMethodsFound>());

		[Test]
		public void InvalidSyntax() =>
			Assert.That(
				Assert.Throws<Type.ParsingFailed>(() => new Type(package, Base.Count, "has log\na b")).
					InnerException, Is.TypeOf<Type.InvalidLine>());

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
			Assert.That(program.Implements[0].Trait.Name, Is.EqualTo(Base.App));
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