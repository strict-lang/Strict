using NUnit.Framework;

namespace Strict.Language.Tests
{
	public class MethodTests
	{
		[SetUp]
		public void CreateType() =>
			type = new Type(new Package(nameof(MethodTests)), nameof(TypeTests), @"has log
method Run
	log.WriteLine");
		private Type type;

		[Test]
		public void MethodMustStartWithMethodOrFrom() =>
			Assert.Throws<Method.InvalidSyntax>(() => new Method(type, "function",new string[0]));

		[Test]
		public void MustMustHaveAName() =>
			Assert.Throws<LineLexer.NoMoreWords>(() => new Method(type, "method", new string[0]));

		[Test]
		public void ParseDefinition()
		{
			var method = new Method(type, "method Run", new string[0]);
			Assert.That(method.Name, Is.EqualTo("Run"));
			Assert.That(method.Parameters, Is.Empty);
			Assert.That(method.ReturnType, Is.EqualTo(type.GetType(Base.None)));
		}

		[Test]
		public void ParseFrom()
		{
			var method = new Method(type, "from(number)", new string[0]);
			Assert.That(method.Name, Is.EqualTo("from"));
			Assert.That(method.Parameters, Has.Count.EqualTo(1), method.Parameters.ToWordString());
			Assert.That(method.Parameters[0].Type, Is.EqualTo(type.GetType("Number")));
			Assert.That(method.ReturnType, Is.EqualTo(type));
		}
	}
}