using NUnit.Framework;

namespace Strict.Language.Tests
{
	public class TypeTests
	{
		[SetUp]
		public void CreatePackage()
		{
			package = new TestPackage();
			new Type(package, Base.App, "Run");
		}

		private Package package;

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
					InnerException, Is.TypeOf<Type.ExtraWhitespacesFoundAtBeginningOfLine>());
			Assert.That(
				Assert.Throws<Type.ParsingFailed>(() =>
					new Type(package, Base.HashCode, "has\t")).InnerException,
				Is.TypeOf<Type.ExtraWhitespacesFoundAtEndOfLine>());
		}

		[Test]
		public void TypeParsersMustStartWithImplementOrHas() =>
			Assert.That(Assert.Throws<Type.ParsingFailed>(() => new Type(package, Base.Count,
					@"Run
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
					InnerException, Is.TypeOf<Method.InvalidSyntax>());

		[Test]
		public void GetUnknownTypeWillCrash() =>
			Assert.Throws<Context.TypeNotFound>(() => package.GetType(Base.Computation));

		[Test]
		public void SimpleApp() =>
			CheckApp(new Type(package, "Program", @"implement App
has log
Run
	log.Write(""Hello World!"")"));

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
Run
	for number in Range(0, 10)
		log.Write(""Counting: "" + number)"));

		[Test]
		public void Trait()
		{
			var app = new Type(package, "DummyApp", "Run");
			Assert.That(app.IsTrait, Is.True);
			Assert.That(app.Name, Is.EqualTo("DummyApp"));
			Assert.That(app.Methods[0].Name, Is.EqualTo("Run"));
		}

		[Test]
		public void FileExtensionMustBeStrict() =>
			Assert.ThrowsAsync<Type.FileExtensionMustBeStrict>(() =>
				new Type(package, "DummyApp", "Run").ParseFile("test.txt"));

		[Test]
		public void FilePathMustMatchPackageName() =>
			Assert.ThrowsAsync<Type.FilePathMustMatchPackageName>(() =>
				new Type(package, "DummyApp", "Run").ParseFile("test.strict"));

		[Test]
		public void FilePathMustMatchMainPackageName() =>
			Assert.ThrowsAsync<Type.FilePathMustMatchPackageName>(() =>
				new Type(new Package(package, nameof(TypeTests)), "DummyApp", "Run").ParseFile(
					nameof(TypeTests) + "\\test.strict"));
	}
}