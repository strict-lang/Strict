using Strict.Expressions;
using Strict.Language;

namespace Strict.HighLevelRuntime;

internal sealed class BodyEvaluator(Executor executor)
{
	public ValueInstance Evaluate(Body body, ExecutionContext ctx, bool runOnlyTests)
	{
		executor.Statistics.BodyCount++;
		if (runOnlyTests)
			inlineTestDepth++;
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
				inlineTestDepth--;
		}
	}

	/// <summary>
	/// Evaluate inline tests at top-level only (outermost call), avoid recursion
	/// </summary>
	internal int inlineTestDepth;

	private ValueInstance TryEvaluate(Body body, ExecutionContext ctx, bool runOnlyTests)
	{
		var last = executor.noneInstance;
		for (var index = 0; index < body.Expressions.Count; index++)
		{
			var e = body.Expressions[index];
			var isTest = !ReferenceEquals(e, body.Expressions[^1]) && IsStandaloneInlineTest(e);
			if (isTest)
				executor.Statistics.TestExpressions++;
			if (isTest == !runOnlyTests && e is not Declaration && e is not MutableReassignment ||
				runOnlyTests && e is Declaration decl && DeclarationReferencesAnyMember(body, decl))
				continue;
			last = executor.RunExpression(e, ctx);
			if (ctx.ExitMethodAndReturnValue.HasValue)
				return ctx.ExitMethodAndReturnValue.Value;
			if (runOnlyTests && isTest)
			{
				if (!last.Boolean)
					throw new Executor.TestFailed(body.Method, e, last, GetTestFailureDetails(e, ctx));
				last = GetStandaloneInlineTestComparedValue(e, ctx) ?? last;
			}
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

	private ValueInstance? GetStandaloneInlineTestComparedValue(Expression expression,
		ExecutionContext ctx) => expression switch
	{
		Binary { Method.Name: BinaryOperator.Is, Instance: Value v } => v.Data,
		Binary { Method.Name: BinaryOperator.Is, Instance: not null } binary =>
			executor.RunExpression(binary.Instance, ctx),
		Not { Instance: Binary { Method.Name: BinaryOperator.Is, Instance: Value v } } => v.Data,
		Not { Instance: Binary { Method.Name: BinaryOperator.Is, Instance: not null } binary } =>
			executor.RunExpression(binary.Instance, ctx), //ncrunch: no coverage
		_ => null
	};

	private static bool ExpressionReferencesMember(Expression expr, string memberName) =>
		expr switch
		{
			MemberCall m => m.Member.Name == memberName,
			MethodCall call =>
				call.Instance != null && ExpressionReferencesMember(call.Instance, memberName) ||
				call.Arguments.Any(a => ExpressionReferencesMember(a, memberName)),
			List list => list.Values.Any(v => ExpressionReferencesMember(v, memberName)),
			Dictionary dict =>
				dict.KeyType.Name.Equals(memberName, StringComparison.OrdinalIgnoreCase) ||
				dict.MappedValueType.Name.Equals(memberName, StringComparison.OrdinalIgnoreCase),
			_ => false
		};

	private static bool IsStandaloneInlineTest(Expression e) =>
		e.ReturnType.IsBoolean && e is not If && e is not Return && e is not Declaration &&
		e is not MutableReassignment;

	private static bool DeclarationReferencesAnyMember(Body body, Declaration decl)
	{
		var members = body.Method.Type.Members;
		for (var i = 0; i < members.Count; i++)
			if (!members[i].IsConstant && ExpressionReferencesMember(decl.Value, members[i].Name))
				return true;
		return false;
	}

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