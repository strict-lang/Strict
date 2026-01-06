namespace Strict.Validators.Tests;

public sealed class VisitorTests
{
	[SetUp]
	public void Setup()
	{
		type = new Type(TestPackage.Instance,
			new TypeLines(nameof(VisitorTests), "has logger", "Run", "\tlogger.Log(5)"));
		parser = new MethodExpressionParser();
		type.ParseMembersAndMethods(parser);
	}

	private Type type = null!;
	private ExpressionParser parser = null!;

	[TearDown]
	public void TearDown() => type.Dispose();

	private sealed class CountingVisitor : Visitor
	{
		public int BodyCount { get; private set; }
		public int IfCount { get; private set; }
		public int ForCount { get; private set; }
		public int MutableReassignmentCount { get; private set; }
		public int ListCallCount { get; private set; }

		protected override void Visit(Body body, object? context = null)
		{
			BodyCount++;
			base.Visit(body, context);
		}

		protected override Expression? Visit(Expression? expression, Body? body,
			object? context = null)
		{
			switch (expression)
			{
			case If:
				IfCount++;
				break;
			case For:
				ForCount++;
				break;
			case MutableReassignment:
				MutableReassignmentCount++;
				break;
			case ListCall:
				ListCallCount++;
				break;
			}
			return base.Visit(expression, body, context);
		}
	}

	[Test]
	public void VisitsInnerBodiesInIfThenAndElse()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\tif true",
			"\t\tlogger.Log(1)",
			"\telse",
			"\t\tlogger.Log(2)"
		]);
		var visitor = new CountingVisitor();
		visitor.Visit(method, true);
		Assert.That(visitor.IfCount, Is.EqualTo(1));
	}

	[Test]
	public void VisitsMutableReassignment()
	{
		var method = new Method(type, 1, parser, [
			"Run(mutable parameter Number)",
			"\tparameter = 5",
			"\t5"
		]);
		var visitor = new CountingVisitor();
		visitor.Visit(method, true);
		Assert.That(visitor.MutableReassignmentCount, Is.EqualTo(1));
	}

	[Test]
	public void VisitsForExpressionAndItsBody()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\tfor Range(2, 5)",
			"\t\tlogger.Log(index)"
		]);
		var visitor = new CountingVisitor();
		visitor.Visit(method, true);
		Assert.That(visitor.ForCount, Is.EqualTo(1));
	}

	[Test]
	public void VisitsListCallListAndIndex()
	{
		var method = new Method(type, 1, parser, [
			"Run",
			"\tconstant numbers = (1, 2, 3)",
			"\tlogger.Log(numbers(1))"
		]);
		var visitor = new CountingVisitor();
		visitor.Visit(method, true);
		Assert.That(visitor.ListCallCount, Is.EqualTo(1));
	}
}