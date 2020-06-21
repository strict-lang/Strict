using NUnit.Framework;

namespace Strict.Language.Tests
{
	public class TypeTests
	{
		[SetUp]
		public void CreatePackage() => package = new Package(nameof(TypeTests));
		private Package package;

		[Test]
		public void EmptyIsNotAllowed() =>
			Assert.Throws<Type.EmptyLine>(() => new Type(package, Base.None, ""));

		[Test]
		public void WhitespacesAreNotAllowed()
		{
			Assert.Throws<Type.ExtraWhitespacesFound>(() => new Type(package, Base.None, " "));
			Assert.Throws<Type.ExtraWhitespacesFound>(() => new Type(package, Base.None, "has\t"));
		}

		[Test]
		public void LineWithOneWordIsNotAllowed() =>
			Assert.Throws<Type.LineWithJustOneWord>(() => new Type(package, Base.None, "has"));

		[Test]
		public void TypeParsersMustStartWithImplementOrHas() =>
			Assert.Throws<Type.TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies>(() =>
				new Type(package, Base.None, @"method Run
	log.WriteLine"));

		[Test]
		public void JustMembersIsNotValidCode() =>
			Assert.Throws<Type.NoMethodsFound>(() => new Type(package, Base.None, @"has log
has count"));

		[Test]
		public void InvalidSyntax() =>
			Assert.Throws<Type.InvalidSyntax>(() => new Type(package, Base.None, "has a\na b"));

		[Test]
		public void SimpleApp()
		{
			var program = new Type(package, "Program", @"implement App
has log
method Run
	log.WriteLine(""Hello World!"")");
			//need FindType implementation first: Assert.That(program.Implements[0].Trait.Name, Is.EqualTo("App"));
			Assert.That(program.Members[0].Name, Is.EqualTo("log"));
			Assert.That(program.Methods[0].Name, Is.EqualTo("Run"));
		}

		[Test]
		public void AnotherApp()
		{
			var program = new Type(package, "Program", @"implement App
has log
method Run
	for number in Range(0, 10)
		log.WriteLine(""Counting: "" + number)");
			//need FindType implementation first: Assert.That(program.Implements[0].Trait.Name, Is.EqualTo("App"));
			Assert.That(program.Members[0].Name, Is.EqualTo("log"));
			Assert.That(program.Methods[0].Name, Is.EqualTo("Run"));
		}

		[Test]
		public void Trait()
		{
			var app = new Type(package, "App", "method Run");
			Assert.That(app.IsTrait, Is.True);
			Assert.That(app.Name, Is.EqualTo("App"));
			Assert.That(app.Methods[0].Name, Is.EqualTo("Run"));
		}
	}
}