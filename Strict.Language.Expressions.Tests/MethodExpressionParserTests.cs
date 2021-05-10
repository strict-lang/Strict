using NUnit.Framework;

namespace Strict.Language.Expressions.Tests
{
    public class MethodExpressionParserTests
    {
        [Test]
        public void GetSingleLine()
        {
            var lines = MethodExpressionParser.GetMainLines("let number = 5");
            Assert.That(lines, Has.Count.EqualTo(1));
            Assert.That(lines[0], Is.EqualTo("let number = 5"));
        }

        [Test]
        public void GetMultipleLines()
        {
            var lines = MethodExpressionParser.GetMainLines(@"let number = 5
let other = 3");
            Assert.That(lines, Has.Count.EqualTo(2));
            Assert.That(lines[0], Is.EqualTo("let number = 5"));
            Assert.That(lines[1], Is.EqualTo("let other = 3"));
        }

        [Test]
        public void GetNestedLines()
        {
            var lines = MethodExpressionParser.GetMainLines(@"let number = 5
if number is 5
	return true
return false");
            Assert.That(lines, Has.Count.EqualTo(3));
            Assert.That(lines[0], Is.EqualTo("let number = 5"));
            Assert.That(lines[1], Is.EqualTo("if number is 5\n\treturn true"));
            Assert.That(lines[2], Is.EqualTo("return false"));
        }
    }
}