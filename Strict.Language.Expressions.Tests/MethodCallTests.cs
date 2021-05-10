using NUnit.Framework;

namespace Strict.Language.Expressions.Tests
{
    public class MethodCallTests : TestExpressions
    {
        [Test]
        public void ParseCall() =>
            ParseAndCheckOutputMatchesInput("log.WriteLine(\"Hi\")",
                new MethodCall(new MemberCall(member), member.Type.Methods[0],
                    new Text(type, "Hi")));
    }
}