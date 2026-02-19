using Strict.Language.Tests;

namespace Strict.Expressions.Tests;

public abstract class TestExpressions : MethodExpressionParser
{
	protected TestExpressions()
	{
		type = new Type(TestPackage.Instance, new TypeLines("dummy", "Run")).
			ParseMembersAndMethods(this);
		type.GetType(Base.Boolean);
		member = new Member(type, "logger", null);
		type.Members.Add(member);
		method = new Method(type, 0, this, [MethodTests.Run]);
		methodWithBody = new Method(type, 0, this, [
			MethodTests.Run, "\tconstant variable = 5", "\tvariable + 5"
		]);
		type.Methods.AddRange(new List<Method> { method, methodWithBody });
		numberFive = new Number(type, 5);
		list = new List(new Body(method), [numberFive]);
		five = new Member(type, "five", type.GetType(Base.Number)) { InitialValue = numberFive };
		type.Members.Add(five);
	}

	protected readonly Type type;
	protected readonly Member member;
	protected readonly Method method;
	protected readonly Method methodWithBody;
	protected readonly Number numberFive;
	protected readonly Member five;
	protected readonly List list;

	[SetUp]
	public void SayNoToConsoles()
	{
		noConsole = new NoConsoleWriteLineAllowed();
		noConsole.SetVirtualConsoleWriter();
	}

	private NoConsoleWriteLineAllowed noConsole = null!;

	[TearDown]
	public void NoConsoleAllowed()
	{
		TestPackage.Instance.Remove(type);
		noConsole.CheckIfConsoleIsEmpty();
	}

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
		new(fromType.FindMethod(Method.From, arguments)!, null, arguments);

	protected static Binary CreateBinary(Expression left, string operatorName, Expression right)
	{
		var arguments = new[] { right };
		return new Binary(left, left.ReturnType.GetMethod(operatorName, arguments), arguments);
	}

	protected Expression GetCondition(bool isNot = false)
	{
		var isExpression = CreateBinary(new MemberCall(null, five), BinaryOperator.Is, numberFive);
		return isNot
			? new Not(TestPackage.Instance.GetType(Base.Boolean).GetMethod(UnaryOperator.Not, []),
				isExpression)
			: isExpression;
	}

	protected Not CreateNot(Expression right) =>
		new(TestPackage.Instance.GetType(Base.Boolean).GetMethod(UnaryOperator.Not, []), right);
}