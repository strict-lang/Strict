using NUnit.Framework;
using Strict.Language.Expressions;
using Strict.Language.Extensions;

namespace Strict.Language.Tests
{
	public class MethodTests
	{
		[SetUp]
		public void CreateType() =>
			type = new Type(new TestPackage(), nameof(TypeTests), @"has log
Run
	log.WriteLine");
		private Type type;

		[Test]
		public void MustMustHaveAName() =>
			Assert.Throws<Method.InvalidSyntax>(() => new Method(type, "a b", new string[0]));

		[Test]
		public void MethodNameCantBeKeyword() =>
			Assert.Throws<Method.MethodNameCantBeKeyword>(
				() => new Method(type, "if", new string[0]));
		
		[Test]
		public void ParametersMustNotBeEmpty() =>
			Assert.Throws<Method.EmptyParametersMustBeRemoved>(() => new Method(type, "a()", new string[0]));

		[Test]
		public void ParseDefinition()
		{
			var method = new Method(type, "Run", new string[0]);
			Assert.That(method.Name, Is.EqualTo("Run"));
			Assert.That(method.Parameters, Is.Empty);
			Assert.That(method.ReturnType, Is.EqualTo(type.GetType(Base.None)));
		}

		[Test]
		public void ParseFrom()
		{
			var method = new Method(type, "from(number)", new string[0]);
			Assert.That(method.Name, Is.EqualTo("from"));
			Assert.That(method.Parameters, Has.Count.EqualTo(1), method.Parameters.ToWordListString());
			Assert.That(method.Parameters[0].Type, Is.EqualTo(type.GetType("Number")));
			Assert.That(method.ReturnType, Is.EqualTo(type));
		}

		[Test]
		public void ParseBody()
		{
			var code = @"log.WriteLine(""Hey"")";
			var method = new Method(type, @"Run", new[] { "\t" + code });
			Assert.That(method.Body.Expressions, Has.Count.EqualTo(1));
			var methodCall = method.Body.Expressions[0] as MethodCall;
			Assert.That(methodCall.ReturnType, Is.EqualTo(type.GetType(Base.None)));
			Assert.That(methodCall.Method.Type, Is.EqualTo(type.GetType(Base.Log)));
			Assert.That(methodCall.Method.Name, Is.EqualTo("WriteLine"));
			Assert.That(methodCall.Arguments.Count, Is.EqualTo(1));
			var text = methodCall.Arguments[0] as Value;
			Assert.That(text.Data, Is.EqualTo("Hey"));
			Assert.That(methodCall.ToString(), Is.EqualTo(code));
		}

		[Test]
		public void ParseValues()
		{
			var code = @"return 5 + true";
			var method = new Method(type, @"Run returns Number", new[] { "\t" + code });
			var returnExpression = method.Body.Expressions[0] as Return;
			Assert.That(returnExpression.ReturnType, Is.EqualTo(type.GetType(Base.Number)));
			var binary = returnExpression.Expression as Binary;
			Assert.That(binary.Left, Is.InstanceOf<Number>());
			Assert.That(binary.Right, Is.InstanceOf<Boolean>());
			Assert.That(returnExpression.ToString(), Is.EqualTo(code));
		}
	}
}