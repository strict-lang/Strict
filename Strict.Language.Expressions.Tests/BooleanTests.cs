using NUnit.Framework;

namespace Strict.Language.Expressions.Tests
{
    public class BooleanTests : TestExpressions
    {
        [Test]
        public void ParseTrue() =>
            ParseAndCheckOutputMatchesInput("true", new Boolean(method, true));

        [Test]
        public void ParseFalse() =>
            ParseAndCheckOutputMatchesInput("false", new Boolean(method, false));
    }
}