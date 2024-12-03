using NUnit.Framework;
using Strict.Language.Tests;

namespace Strict.Language.Expressions.Tests;

public abstract class TestExpressions : MethodExpressionParser
{
	private sealed class Data
	{
		public readonly Type type;
		public readonly Type boolean;
		public readonly Member member;
		public readonly Method method;
		public readonly Method methodWithBody;
		public readonly Number number;
		public readonly Member bla;
		public readonly List list;

		public Data(TestExpressions parent)
		{
			type = new Type(new TestPackage(), new TypeLines("dummy", "Run")).
				ParseMembersAndMethods(parent);
			boolean = type.GetType(Base.Boolean);
			member = new Member(type, "log", null);
			type.Members.Add(member);
			method = new Method(type, 0, parent, [MethodTests.Run]);
			methodWithBody = new Method(type, 0, parent, [
				MethodTests.Run, "\tconstant variable = 5", "\tvariable + 5"
			]);
			type.Methods.AddRange(new List<Method> { method, methodWithBody });
			number = new Number(type, 5);
			list = new List(new Body(method), [new Number(type, 5)]);
			bla = new Member(type, "bla", number);
			type.Members.Add(bla);
		}
	}

	private Data? testData; // Will be set during SetUp-method
	private Data TestData => testData!;

	protected Type type => TestData.type;
	protected Type boolean => TestData.boolean;
	protected Member member => TestData.member;
	protected Method method => TestData.method;
	protected Method methodWithBody => TestData.methodWithBody;
	protected Number number => TestData.number;
	protected Member bla => TestData.bla;
	protected List list => TestData.list;

	[SetUp]
	public void SayNoToConsoles()
	{
		noConsole = new NoConsoleWriteLineAllowed();
		noConsole.SetVirtualConsoleWriter();
	}

	[SetUp]
	public void CreateData() => testData = new Data(this);

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

	protected static MethodCall
		CreateFromMethodCall(Type fromType, params Expression[] arguments) =>
		new(fromType.FindMethod(Method.From, arguments, new MethodExpressionParser())!, null,
			arguments);

	protected static Binary CreateBinary(Expression left, string operatorName, Expression right)
	{
		var arguments = new[] { right };
		return new Binary(left,
			left.ReturnType.GetMethod(operatorName, arguments, new MethodExpressionParser()),
			arguments);
	}

	protected Binary GetCondition(bool isNot = false) =>
		CreateBinary(new MemberCall(null, bla), isNot
			? BinaryOperator.IsNot
			: BinaryOperator.Is, number);
}