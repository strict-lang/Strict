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
				Environment.NewLine + body.Expressions.ToWordList(Environment.NewLine), ex);
		}
		finally
		{
			if (runOnlyTests)
				executor.DecrementInlineTestDepth();
		}
	}

	private ValueInstance TryEvaluate(Body body, ExecutionContext ctx, bool runOnlyTests)
	{
		var last = executor.None(ctx.This?.ReturnType ?? body.Method.Type);
		foreach (var e in body.Expressions)
		{
			var isTest = !e.Equals(body.Expressions[^1]) && IsStandaloneInlineTest(e);
			if (isTest)
				executor.Statistics.TestsCount++;
			if (isTest == !runOnlyTests && e is not Declaration && e is not MutableReassignment ||
				runOnlyTests && e is Declaration &&
				body.Method.Type.Members.Any(m => !m.IsConstant && e.ToString().Contains(m.Name)))
				continue;
			last = executor.RunExpression(e, ctx);
			if (ctx.ExitMethodAndReturnValue.HasValue)
				return ctx.ExitMethodAndReturnValue.Value;
			if (runOnlyTests && isTest && !Executor.ToBool(last))
				throw new Executor.TestFailed(body.Method, e, last, GetTestFailureDetails(e, ctx));
		}
		if (runOnlyTests && last.Value == null && body.Method.Name != Base.Run &&
			body.Expressions.Count > 1)
			throw new Executor.MethodRequiresTest(body.Method, body);
		if (runOnlyTests || last.ReturnType.IsError || body.Method.ReturnType == last.ReturnType)
			return last;
		if (body.Method.ReturnType.IsMutable && !last.ReturnType.IsMutable &&
			last.ReturnType == ((GenericTypeImplementation)body.Method.ReturnType).
			ImplementationTypes[0])
			return executor.CreateValueInstance(body.Method.ReturnType, last.Value);
		throw new Executor.ReturnTypeMustMatchMethod(body, last);
	}

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