using NUnit.Framework;

namespace Strict.Language.Expressions.Tests
{
	public class AssignmentTests : TestExpressions
	{
		[Test]
		public void MultipleStatementsAreNotAllowedHere() =>
			Assert.That(() => ParseExpression(method, "let number = 5\nlet other = 3"),
				Throws.Exception.InstanceOf<MultipleExpressionsGiven>());

		[Test]
		public void ParseNumber() =>
			Assert.That(ParseExpression(method, "let number = 5"),
				Is.EqualTo(new Assignment(new Identifier(nameof(number), number.ReturnType),
					number)));

		[Test]
		public void AssignmentToString()
		{
			var input = "let value = 1";
			Assert.That(ParseExpression(method, input).ToString(), Is.EqualTo(input));
		}

		[Test]
		public void AssignmentGetHashCode()
		{
			var assignment = ParseExpression(method, "let value = 1") as Assignment;
			Assert.That(assignment.GetHashCode(),
				Is.EqualTo(assignment.Name.GetHashCode() ^ assignment.Value.GetHashCode()));
		}

		[Test]
		public void LetWithoutVariableNameCannotParse() =>
			Assert.That(() => ParseExpression(method, "let 5"),
				Throws.Exception.InstanceOf<Assignment.IncompleteLet>());

		[Test]
		public void LetWithoutExpressionCannotParse() =>
			Assert.That(() => ParseExpression(method, "let value = abc"),
				Throws.Exception.InstanceOf<Assignment.InvalidExpression>());
	}
}