using System.Linq;
using NUnit.Framework;

namespace Strict.Language.Tests
{
	public class TypeParserTests
	{
		[SetUp]
		public void CreateParser() => parser = new TypeParser(new Context());
		private TypeParser parser;

		[Test]
		public void EmptyIsNotAllowed() =>
			Assert.Throws<TypeParser.EmptyLine>(() => parser.ParseCode("Empty", ""));

		[Test]
		public void WhitespacesAreNotAllowed()
		{
			Assert.Throws<TypeParser.ExtraWhitespacesFound>(() => parser.ParseCode("", " "));
			Assert.Throws<TypeParser.ExtraWhitespacesFound>(() => parser.ParseCode("", "has\t"));
		}

		[Test]
		public void LineWithOneWordIsNotAllowed() =>
			Assert.Throws<TypeParser.LineWithJustOneWord>(() => parser.ParseCode("", "has"));

		[Test]
		public void TypeParsersMustStartWithImplementOrHas() =>
			Assert.Throws<TypeParser.MustStartWithImplementOrHas>(() => parser.ParseCode("", @"method Run()"));

		[Test]
		public void JustMembersIsntValidCode() =>
			Assert.Throws<TypeParser.NoMethodsFound>(() => parser.ParseCode("", @"has log
has count"));
		
		[Test]
		public void InvalidSyntax() =>
			Assert.Throws<TypeParser.InvalidSyntax>(() => parser.ParseCode("", "has a\na b"));

		[Test]
		public void SimpleApp()
		{
			var app = parser.ParseCode("Program", @"implement App
has log
method Run()
	log.WriteLine(""Hello World!"")");
			Assert.That(app.Implement?.Trait.Name, Is.EqualTo("App"));
			Assert.That(app.Has.First().Name, Is.EqualTo("log"));
			Assert.That(app.Methods.First().Name, Is.EqualTo("Run()"));//TODO: remove (), need method parser
			//void doesn't need a test, has no side effects inside strict: log.LastLine() is ""Hello World!""
		}
	}
}