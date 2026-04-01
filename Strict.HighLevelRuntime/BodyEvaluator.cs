using Strict.Expressions;
using Strict.Language;

namespace Strict.HighLevelRuntime;

internal sealed class BodyEvaluator(Interpreter interpreter)
{
	public ValueInstance Evaluate(Body body, ExecutionContext ctx, bool runOnlyTests)
	{
		interpreter.Statistics.BodyCount++;
		if (runOnlyTests)
			inlineTestDepth.Value++;
		try
		{
			return TryEvaluate(body, ctx, runOnlyTests);
		}
		catch (InterpreterExecutionFailed ex)
		{
			//TODO: is this really needed, can't we build the failure message already correct where the original InterpreterExecutionFailed is thrown?
			if (ex.Message.Contains(InterpreterExecutionFailed.GetMethodFailureHeader(body.Method),
				StringComparison.Ordinal))
				throw;
			var fileLineNumber = ex.MethodName == body.Method.ToString()
				? ex.FileLineNumber
				: ctx.CurrentExpressionLineNumber;
			throw new InterpreterExecutionFailed(body.Method, fileLineNumber,
				InterpreterExecutionFailed.BuildMethodFailureMessage(body.Method, fileLineNumber,
					body.Expressions, ex.Headline), ex, false);
		}
		finally
		{
			if (runOnlyTests)
				inlineTestDepth.Value--;
		}
	}

	/// <summary>
	/// Evaluate inline tests at top-level only (outermost call), avoid recursion
	/// </summary>
	internal int InlineTestDepth => inlineTestDepth.Value;
	private readonly ThreadLocal<int> inlineTestDepth = new(() => 0);

	private ValueInstance TryEvaluate(Body body, ExecutionContext ctx, bool runOnlyTests)
	{
		var last = interpreter.noneInstance;
		var count = body.Expressions.Count;
		HashSet<string>? skippedVariables = null;
		var pastTestBlock = false;
		for (var index = 0; index < count; index++)
		{
			var e = body.Expressions[index];
			var isTest = !pastTestBlock && index < count - 1 && IsStandaloneInlineTest(e);
			if (!isTest && e is not Declaration && e is not MutableReassignment)
				pastTestBlock = true;
			if (isTest)
				interpreter.Statistics.TestExpressions++;
			if (isTest == !runOnlyTests && e is not Declaration && e is not MutableReassignment
					&& e is not For ||
					runOnlyTests && e is Declaration decl && (DeclarationReferencesAnyMember(body, decl) ||
					skippedVariables != null &&
					ExpressionReferencesSkippedVariable(decl.Value, skippedVariables)) ||
					runOnlyTests && skippedVariables != null && e is not Declaration &&
					ExpressionReferencesSkippedVariable(e, skippedVariables) ||
					runOnlyTests && e is For forExpr && ForExpressionReferencesAnyMember(body, forExpr))
			{
				if (runOnlyTests && e is Declaration skippedDecl)
					(skippedVariables ??= []).Add(skippedDecl.Name);
				continue;
			}
			ctx.CurrentExpressionLineNumber = e.LineNumber;
			last = interpreter.RunExpression(e, ctx);
			if (ctx.ExitMethodAndReturnValue.HasValue)
				return ctx.ExitMethodAndReturnValue.Value;
			if (runOnlyTests && isTest && !last.Boolean)
				throw new Interpreter.TestFailed(body.Method, e, last, GetTestFailureDetails(e, ctx));
		}
		if (runOnlyTests && count > 1 && last.Equals(interpreter.noneInstance) &&
			body.Method.Name != Method.Run)
			throw new Interpreter.MethodRequiresTest(body.Method, body);
		if (runOnlyTests || last.IsError || last.IsType(body.Method.ReturnType))
			return last;
		if (body.Method.ReturnType.IsMutable && !last.IsMutable &&
			last.IsType(((GenericTypeImplementation)body.Method.ReturnType).ImplementationTypes[0]))
			return new ValueInstance(last, body.Method.ReturnType);
		throw new Interpreter.ReturnTypeMustMatchMethod(body, last);
	}

	private static bool ExpressionReferencesMember(Expression expr, string memberName) =>
		expr switch
		{
			MemberCall m => m.Member.Name == memberName && m.Instance == null,
			ListCall lc => ExpressionReferencesMember(lc.List, memberName),
			MethodCall call => call.Instance == null && call.Method.Name != Method.From ||
				call.Instance != null && ExpressionReferencesMember(call.Instance, memberName) ||
				call.Arguments.Any(a => ExpressionReferencesMember(a, memberName)),
			List list => list.Values.Any(v => ExpressionReferencesMember(v, memberName)),
			Dictionary dict =>
				dict.KeyType.Name.Equals(memberName, StringComparison.OrdinalIgnoreCase) ||
				dict.MappedValueType.Name.Equals(memberName, StringComparison.OrdinalIgnoreCase),
			_ => false
		};

	private static bool ExpressionReferencesSkippedVariable(Expression expr,
		IReadOnlySet<string> skippedVariables) =>
		expr switch
		{
			VariableCall v => skippedVariables.Contains(v.Variable.Name),
			ParameterCall p => skippedVariables.Contains(p.Parameter.Name),
			MethodCall call => call.Instance != null &&
				ExpressionReferencesSkippedVariable(call.Instance, skippedVariables) ||
				call.Arguments.Any(a => ExpressionReferencesSkippedVariable(a, skippedVariables)),
			MemberCall m => m.Instance != null &&
				ExpressionReferencesSkippedVariable(m.Instance, skippedVariables),
			List list => list.Values.Any(v => ExpressionReferencesSkippedVariable(v, skippedVariables)),
			For f => ExpressionReferencesSkippedVariable(f.Iterator, skippedVariables) ||
				BodyExpressionsReferenceSkippedVariable(f.Body, skippedVariables),
			If iff => ExpressionReferencesSkippedVariable(iff.Condition, skippedVariables) ||
				BodyExpressionsReferenceSkippedVariable(iff.Then, skippedVariables) ||
				iff.OptionalElse != null &&
				BodyExpressionsReferenceSkippedVariable(iff.OptionalElse, skippedVariables),
			MutableReassignment mr =>
				ExpressionReferencesSkippedVariable(mr.Target, skippedVariables) ||
				ExpressionReferencesSkippedVariable(mr.Value, skippedVariables),
			Body body => body.Expressions.Any(e =>
				ExpressionReferencesSkippedVariable(e, skippedVariables)),
			_ => false
		};

	private static bool BodyExpressionsReferenceSkippedVariable(Expression expr,
		IReadOnlySet<string> skippedVariables) =>
		expr is Body body
			? body.Expressions.Any(e => ExpressionReferencesSkippedVariable(e, skippedVariables))
			: ExpressionReferencesSkippedVariable(expr, skippedVariables);

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

	private static bool ForExpressionReferencesAnyMember(Body body, For forExpr)
	{
		var members = body.Method.Type.Members;
		for (var i = 0; i < members.Count; i++)
		{
			var name = members[i].Name;
			if (!members[i].IsConstant && (ExpressionReferencesMember(forExpr.Iterator, name) ||
				BodyExpressionsReferencesMember(forExpr.Body, name)))
				return true;
		}
		return false;
	}

	private static bool BodyExpressionsReferencesMember(Expression expr, string memberName) =>
		expr is Body body
			? body.Expressions.Any(e => ExpressionReferencesMember(e, memberName))
			: ExpressionReferencesMember(expr, memberName);

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
		interpreter.RunExpression(binary.Instance!, ctx) + " " + op + " " +
		interpreter.RunExpression(binary.Arguments[0], ctx);
}