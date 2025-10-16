using System.Globalization;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public sealed class Executor(Package basePackage, bool evaluateInlineTests = false)
{
	// Evaluate inline tests at top-level only (outermost call), avoid recursion
	private readonly bool evaluateInlineTests = evaluateInlineTests;
	private int inlineTestDepth;

	public ValueInstance Execute(Method method, ValueInstance? instance,
		IReadOnlyList<ValueInstance> args)
	{
		var context = new ExecutionContext { This = instance };
		for (var i = 0; i < method.Parameters.Count; i++)
		{
			var param = method.Parameters[i];
			var arg = i < args.Count
				? args[i]
				: throw new ArgumentException("Missing argument: " + param.Name);
			context.Set(param.Name, arg);
		}
		try
		{
			var bodyExpr = method.GetBodyAndParseIfNeeded();
			var result = bodyExpr is Body body
				? EvaluateBody(body, context, isTopLevel: true)
				: Evaluate(bodyExpr, context);
			return result;
		}
		catch (ReturnSignal ret)
		{
			return ret.Value;
		}
	}

	public ValueInstance Evaluate(Expression expr, ExecutionContext context) =>
		expr switch
		{
			Body body => EvaluateBody(body, context, isTopLevel: false),
			Value v => new ValueInstance(v.ReturnType, v.Data),
			ParameterCall or VariableCall => context.Get(expr.ToString()),
			MemberCall m => EvaluateMember(m, context),
			If iff => EvaluateIf(iff, context),
			Return r => EvaluateReturn(r, context),
			MethodCall call => EvaluateMethodCall(call, context),
			Declaration c => EvaluateAndAssign(c.Name, c.Value, context),
			MutableReassignment a => EvaluateAndAssign(a.Name, a.Value, context),
			_ => throw new NotSupportedException($"Expression not supported yet: {expr.GetType().Name}")
		};

	private ValueInstance EvaluateBody(Body body, ExecutionContext ctx, bool isTopLevel)
	{
		ValueInstance last =
			new((ctx.This?.ReturnType.Package ?? body.Method.Type.Package).FindType(Base.None)!, null);
		var shouldRunInlineTestsNow = evaluateInlineTests && inlineTestDepth == 0 && isTopLevel;
		if (shouldRunInlineTestsNow)
			inlineTestDepth++;
		try
		{
			foreach (var e in body.Expressions)
			{
				// Skip inline tests by default to avoid infinite recursion
				if (!shouldRunInlineTestsNow &&
					IsStandaloneInlineTest(e))
					continue;
				last = e switch
				{
					Declaration c => EvaluateAndAssign(c.Name, c.Value, ctx),
					MutableReassignment a => EvaluateAndAssign(a.Name, a.Value, ctx),
					_ => Evaluate(e, ctx)
				};
			}
			return last;
		}
		finally
		{
			if (shouldRunInlineTestsNow)
				inlineTestDepth--;
		}
	}

	// An inline test is typically a standalone boolean expression at the method top level,
	// not part of control flow or assignment.
	private static bool IsStandaloneInlineTest(Expression e) =>
		e.ReturnType.Name == Base.Boolean &&
		e is not If &&
		e is not Return &&
		e is not Declaration &&
		e is not MutableReassignment;

	private ValueInstance EvaluateAndAssign(string name, Expression value, ExecutionContext ctx) =>
		ctx.Set(name, Evaluate(value, ctx));

	private ValueInstance EvaluateReturn(Return r, ExecutionContext ctx) =>
		throw new ReturnSignal(Evaluate(r.Value, ctx));

	private ValueInstance EvaluateIf(If iff, ExecutionContext ctx)
	{
		var cond = Evaluate(iff.Condition, ctx);
		var isTrue = ToBool(cond);
		if (isTrue)
			return iff.Then is Body thenBody
				? EvaluateBody(thenBody, ctx, isTopLevel: false)
				: Evaluate(iff.Then, ctx);
		return iff.OptionalElse != null
			? Evaluate(iff.OptionalElse, ctx)
			: new ValueInstance(ctx.This?.ReturnType ?? iff.ReturnType, null);
	}

	private ValueInstance EvaluateMember(MemberCall member, ExecutionContext ctx) =>
		member.Instance == null
			? ctx.Get(member.Member.Name)
			: member.Member.InitialValue != null
				? Evaluate(member.Member.InitialValue, ctx)
				: ctx.Get(member.Member.Name);

	private ValueInstance EvaluateMethodCall(MethodCall call, ExecutionContext ctx)
	{
		var op = call.Method.Name;
		if (IsArithmetic(op) || IsCompare(op))
			return EvaluateArithmeticOrCompare(call, ctx);
		var instance = call.Instance != null
			? Evaluate(call.Instance, ctx)
			: ctx.This;
		var args = new List<ValueInstance>(call.Arguments.Count);
		foreach (var a in call.Arguments)
			args.Add(Evaluate(a, ctx));
		return Execute(call.Method, instance, args);
	}

	private static bool IsArithmetic(string name) =>
		name is BinaryOperator.Plus or BinaryOperator.Minus or BinaryOperator.Multiply or BinaryOperator.Divide or BinaryOperator.Modulate;

	private static bool IsCompare(string name) =>
		name is BinaryOperator.Greater or BinaryOperator.Smaller or BinaryOperator.Is;

	private ValueInstance EvaluateArithmeticOrCompare(MethodCall call, ExecutionContext ctx)
	{
		if (call.Instance == null || call.Arguments.Count != 1)
			throw new InvalidOperationException("Binary call must have instance and 1 argument");
		var left = Evaluate(call.Instance, ctx).Value;
		var right = Evaluate(call.Arguments[0], ctx).Value;
		var op = call.Method.Name;
		if (IsArithmetic(op))
		{
			var l = NumberToDouble(left);
			var r = NumberToDouble(right);
			return op switch
			{
				BinaryOperator.Plus => Number(l + r),
				BinaryOperator.Minus => Number(l - r),
				BinaryOperator.Multiply => Number(l * r),
				BinaryOperator.Divide => Number(l / r),
				BinaryOperator.Modulate => Number(l % r),
				_ => throw new NotSupportedException("Operator not supported for Number: " + op)
			};
		}

		// Compare
		return op switch
		{
			BinaryOperator.Greater => Bool(NumberToDouble(left) > NumberToDouble(right)),
			BinaryOperator.Smaller => Bool(NumberToDouble(left) < NumberToDouble(right)),
			BinaryOperator.Is => Bool(Equals(left, right)),
			_ => throw new NotSupportedException("Operator not supported for Boolean: " + op)
		};
	}

	private static bool ToBool(ValueInstance v) =>
		v.Value switch
		{
			bool b => b,
			Value val when val.ReturnType.Name == Base.Boolean && val.Data is bool bv => bv,
			_ => throw new InvalidOperationException("Expected Boolean, got: " + v)
		};

	private ValueInstance Number(double n) => new(numberType, n);
	private readonly Type numberType = basePackage.FindType(Base.Number)!;
	private ValueInstance Bool(bool b) => new(boolType, b);
	private readonly Type boolType = basePackage.FindType(Base.Boolean)!;

	private static double NumberToDouble(object? n) =>
		Convert.ToDouble(n, CultureInfo.InvariantCulture);

	private sealed class ReturnSignal(ValueInstance value) : Exception
	{
		public ValueInstance Value { get; } = value;
	}
}