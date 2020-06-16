using System.Linq;
using NUnit.Framework;

namespace Strict.Language.Tests
{
	public class CodeFileTests
	{
		[Test]
		public void EmptyIsNotAllowed() =>
			Assert.Throws<CodeFile.EmptyLine>(() => CodeFile.FromCode(""));

		[Test]
		public void WhitespacesAreNotAllowed()
		{
			Assert.Throws<CodeFile.ExtraWhitespacesFound>(() => CodeFile.FromCode(" "));
			Assert.Throws<CodeFile.ExtraWhitespacesFound>(() => CodeFile.FromCode("has\t"));
		}

		[Test]
		public void LineWithOneWordIsNotAllowed() =>
			Assert.Throws<CodeFile.LineWithJustOneWord>(() => CodeFile.FromCode("has"));

		[Test]
		public void CodeFilesMustStartWithImplementOrHas() =>
			Assert.Throws<CodeFile.MustStartWithImplementOrHas>(() => CodeFile.FromCode(@"method Run()"));

		[Test]
		public void JustMembersIsntValidCode() =>
			Assert.Throws<CodeFile.NoMethodsFound>(() => CodeFile.FromCode(@"has log
has count"));
		
		[Test]
		public void InvalidSyntax() =>
			Assert.Throws<CodeFile.InvalidSyntax>(() => CodeFile.FromCode("has a\na b"));

		[Test]
		public void SimpleApp()
		{
			var app = CodeFile.FromCode(@"implement App
has log
method Run()
	log.WriteLine(""Hello World!"")");
			Assert.That(app.Implement?.Name, Is.EqualTo("App"));
			Assert.That(app.Has.First().Name, Is.EqualTo("log"));
			Assert.That(app.Methods.First().Name, Is.EqualTo("Run()"));//TODO: remove (), need method parser
			//void doesn't need a test, has no side effects inside strict: log.LastLine() is ""Hello World!""
		}
	}
}