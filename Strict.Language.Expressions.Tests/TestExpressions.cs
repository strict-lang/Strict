using NUnit.Framework;
using Strict.Language.Tests;

namespace Strict.Language.Expressions.Tests;

public abstract class TestExpressions : MethodExpressionParser
{
	protected TestExpressions()
	{
		type = new Type(new TestPackage(), new TypeLines("dummy", "Run")).
			ParseMembersAndMethods(this);
		boolean = type.GetType(Base.Boolean);
		member = new Member(type, "log", null);
		type.Members.Add(member);
		method = new Method(type, 0, this, [MethodTests.Run]);
		methodWithBody = new Method(type, 0, this, [
			MethodTests.Run, "\tconstant variable = 5", "\tvariable + 5"
		]);
		type.Methods.AddRange(new List<Method> { method, methodWithBody });
		number = new Number(type, 5);
		list = new List(new Body(method), [new Number(type, 5)]);
		bla = new Member(type, "bla", number);
		type.Members.Add(bla);
	}

	protected readonly Type type;
	protected readonly Type boolean;
	protected readonly Member member;
	protected readonly Method method;
	protected readonly Method methodWithBody;
	protected readonly Number number;
	protected readonly Member bla;
	protected readonly List list;

	[SetUp]
	public void SayNoToConsoles()
	{
		noConsole = new NoConsoleWriteLineAllowed();
		noConsole.SetVirtualConsoleWriter();
	}

	private NoConsoleWriteLineAllowed noConsole = null!;

	[TearDown]
	public void NoConsoleAllowed() => noConsole.CheckIfConsoleIsEmpty();

	public void ParseAndCheckOutputMatchesInput(string singleLine, Expression expectedExpression) =>
		ParseAndCheckOutputMatchesInput([singleLine], expectedExpression);

	public void ParseAndCheckOutputMatchesInput(string[] lines, Expression expectedExpression)
	{
		var expression = ParseExpression(lines);
		Assert.That(expression, Is.EqualTo(expectedExpression));
		Assert.That(string.Join(Environment.NewLine, lines), Does.StartWith(expression.ToString()));
	}

	public Expression ParseExpression(params string[] lines)
	{
		var methodLines = new string[lines.Length + 1];
		methodLines[0] = MethodTests.Run;
		for (var index = 0; index < lines.Length; index++)
			methodLines[index + 1] = '\t' + lines[index];
		return new Method(type, 0, this, methodLines).GetBodyAndParseIfNeeded();
	}

	protected static MethodCall CreateFromMethodCall(Type fromType, params Expression[] arguments) =>
		new(fromType.FindMethod(Method.From, arguments, new MethodExpressionParser())!, null, arguments);

	protected static Binary CreateBinary(Expression left, string operatorName, Expression right)
	{
		var arguments = new[] { right };
		return new Binary(left, left.ReturnType.GetMethod(operatorName, arguments, new MethodExpressionParser()), arguments);
	}

	protected Binary GetCondition(bool isNot = false) =>
		CreateBinary(new MemberCall(null, bla), isNot
			? BinaryOperator.IsNot
			: BinaryOperator.Is, number);
}