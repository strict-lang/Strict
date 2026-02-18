using Strict.Expressions;
using Strict.Language;

namespace Strict.HighLevelRuntime;

internal sealed class BodyEvaluator(Executor executor)
{
	public ValueInstance Evaluate(Body body, ExecutionContext ctx, bool runOnlyTests)
	{
		ValueInstance last =
			new((ctx.This?.ReturnType.Package ?? body.Method.Type.Package).FindType(Base.None)!, null);
		if (runOnlyTests)
			executor.IncrementInlineTestDepth();
		try
		{
			foreach (var e in body.Expressions)
			{
				var isTest = !e.Equals(body.Expressions[^1]) && IsStandaloneInlineTest(e);
				if (isTest == !runOnlyTests && e is not Declaration && e is not MutableReassignment ||
					runOnlyTests && e is Declaration &&
					body.Method.Type.Members.Any(m => !m.IsConstant && e.ToString().Contains(m.Name)))
					continue;
				last = executor.RunExpression(e, ctx);
				if (runOnlyTests && isTest && !Executor.ToBool(last))
					throw new Executor.TestFailed(body.Method, e, last, GetTestFailureDetails(e, ctx));
			}
			if (runOnlyTests && last.Value == null && body.Method.Name != Base.Run &&
				body.Expressions.Count > 1)
				throw new Executor.MethodRequiresTest(body.Method, body);
			return runOnlyTests || last.ReturnType.IsError || body.Method.ReturnType == last.ReturnType
				? last
				: body.Method.ReturnType.Name == Base.Character && last.ReturnType.Name == Base.Number
					? new ValueInstance(body.Method.ReturnType, last.Value)
					: body.Method.ReturnType.IsMutable &&
					ctx.This?.ReturnType is GenericTypeImplementation { Generic.Name: Base.Dictionary } &&
					last.ReturnType is GenericTypeImplementation { Generic.Name: Base.List }
						or GenericTypeImplementation
						{
							Generic.Name: Base.Mutable,
							ImplementationTypes:
							[
								GenericTypeImplementation { Generic.Name: Base.List }
							]
						}
						? new ValueInstance(body.Method.ReturnType, ctx.This.Value)
						: body.Method.ReturnType.IsMutable && !last.ReturnType.IsMutable && last.ReturnType ==
						((GenericTypeImplementation)body.Method.ReturnType).ImplementationTypes[0]
							? new ValueInstance(body.Method.ReturnType, last.Value)
							: throw new Executor.ReturnTypeMustMatchMethod(body, last);
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

	private static bool IsStandaloneInlineTest(Expression e) =>
		e.ReturnType.Name == Base.Boolean && e is not If && e is not Return && e is not Declaration &&
		e is not MutableReassignment;

	private string GetTestFailureDetails(Expression expression, ExecutionContext ctx) =>
		expression is Binary { Method.Name: BinaryOperator.Is, Instance: not null } binary &&
		binary.Arguments.Count == 1
			? GetBinaryComparisonDetails(binary, ctx, BinaryOperator.Is)
			: expression is Not { Instance: Binary { Method.Name: BinaryOperator.Is } notBinary } &&
			notBinary.Arguments.Count == 1
				? GetBinaryComparisonDetails(notBinary, ctx, "is not")
				: string.Empty;

	private string GetBinaryComparisonDetails(MethodCall binary, ExecutionContext ctx, string op) =>
		executor.RunExpression(binary.Instance!, ctx) + " " + op + " " +
		executor.RunExpression(binary.Arguments[0], ctx);
}