using Strict.Expressions;
using Strict.Language;

namespace Strict.HighLevelRuntime;

internal sealed class BodyEvaluator(Executor executor)
{
	public ValueInstance Evaluate(Body body, ExecutionContext ctx, bool runOnlyTests)
	{
		executor.Statistics.BodyCount++;
		if (runOnlyTests)
			executor.IncrementInlineTestDepth();
		try
		{
			return TryEvaluate(body, ctx, runOnlyTests);
		}
		catch (ExecutionFailed ex)
		{
			throw new ExecutionFailed(body.Method,
				"Failed in \"" + body.Method.Type.FullName + "." + body.Method.Name + "\":" +
				Environment.NewLine + string.Join(Environment.NewLine, body.Expressions), ex);
		}
		finally
		{
			if (runOnlyTests)
				executor.DecrementInlineTestDepth();
		}
	}

	private ValueInstance TryEvaluate(Body body, ExecutionContext ctx, bool runOnlyTests)
	{
		var last = executor.noneInstance;
		foreach (var e in body.Expressions)
		{
			var isTest = !e.Equals(body.Expressions[^1]) && IsStandaloneInlineTest(e);
			if (isTest)
				executor.Statistics.TestExpressions++;
			if (isTest == !runOnlyTests && e is not Declaration && e is not MutableReassignment ||
				runOnlyTests && e is Declaration decl &&
				body.Method.Type.Members.Any(m => !m.IsConstant && ExpressionReferencesMember(decl.Value, m.Name)))
				continue;
			last = executor.RunExpression(e, ctx);
			if (ctx.ExitMethodAndReturnValue.HasValue)
				return ctx.ExitMethodAndReturnValue.Value;
			if (runOnlyTests && isTest && !last.Boolean)
				throw new Executor.TestFailed(body.Method, e, last, GetTestFailureDetails(e, ctx));
		}
		if (runOnlyTests && last.Equals(executor.noneInstance) && body.Method.Name != Method.Run &&
			body.Expressions.Count > 1)
			throw new Executor.MethodRequiresTest(body.Method, body);
		if (runOnlyTests || last.IsError || last.IsType(body.Method.ReturnType))
			return last;
		if (body.Method.ReturnType.IsMutable && !last.IsMutable &&
			last.IsType(((GenericTypeImplementation)body.Method.ReturnType).ImplementationTypes[0]))
			return new ValueInstance(last, body.Method.ReturnType);
		throw new Executor.ReturnTypeMustMatchMethod(body, last);
	}

	private static bool ExpressionReferencesMember(Expression expr, string memberName) =>
		expr switch
		{
			MemberCall m => m.Member.Name == memberName,
			MethodCall call =>
				call.Instance != null && ExpressionReferencesMember(call.Instance, memberName) ||
				call.Arguments.Any(a => ExpressionReferencesMember(a, memberName)),
			List list => list.Values.Any(v => ExpressionReferencesMember(v, memberName)),
			_ => false
		};

	private static bool IsStandaloneInlineTest(Expression e) =>
		e.ReturnType.IsBoolean && e is not If && e is not Return && e is not Declaration &&
		e is not MutableReassignment;

	private string GetTestFailureDetails(Expression expression, ExecutionContext ctx) =>
		expression is Binary
		{
			Method.Name: BinaryOperator.Is, Instance: not null, Arguments.Count: 1
		} binary
			? GetBinaryComparisonDetails(binary, ctx, BinaryOperator.Is)
			: expression is Not { Instance: Binary { Method.Name: BinaryOperator.Is } notBinary } &&
			notBinary.Arguments.Count == 1
				? GetBinaryComparisonDetails(notBinary, ctx, "is not")
				: string.Empty;

	private string GetBinaryComparisonDetails(MethodCall binary, ExecutionContext ctx, string op) =>
		executor.RunExpression(binary.Instance!, ctx) + " " + op + " " +
		executor.RunExpression(binary.Arguments[0], ctx);
}