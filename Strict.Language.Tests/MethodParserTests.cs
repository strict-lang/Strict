using NUnit.Framework;

namespace Strict.Language.Tests
{
	public class MethodParserTests
	{
		[SetUp]
		public void CreateParser()
		{
			context = new Context();
			parser = new MethodParser(context);
		}

		private Context context;
		private MethodParser parser;

		[Test]
		public void MethodMustStartWithMethodOrFrom() =>
			Assert.Throws<MethodParser.InvalidSyntax>(() => parser.Parse(@"function"));

		[Test]
		public void MustMustHaveAName() =>
			Assert.Throws<LineLexer.NoMoreWords>(() => parser.Parse(@"method"));

		[Test]
		public void ParseDefinition()
		{
			var method = parser.Parse(@"method Run");
			Assert.That(method.Name, Is.EqualTo("Run"));
			Assert.That(method.Parameters, Is.Empty);
			Assert.That(method.ReturnType, Is.EqualTo(Type.Void));
		}

		[Test]
		public void ParseFrom()
		{
			var method = parser.Parse(@"from(number)");
			Assert.That(method.Name, Is.EqualTo("from"));
			Assert.That(method.Parameters, Has.Count.EqualTo(1), method.Parameters.ToWordString());
			Assert.That(method.Parameters[0].Type, Is.EqualTo(context.FindType("Number")));
			Assert.That(method.ReturnType, Is.EqualTo(context.ParentType));
		}
	}
}