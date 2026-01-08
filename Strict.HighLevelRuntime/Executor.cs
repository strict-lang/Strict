using System.Globalization;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public sealed class Executor(Package basePackage, bool evaluateInlineTests = true)
{
	public ValueInstance Execute(Method method, ValueInstance? instance,
		IReadOnlyList<ValueInstance> args)
	{
		if (evaluateInlineTests && validatedMethods.Add(method))
			Execute(method, instance, args, true);
		//TODO: only execute test logic from TestExecutor, not the non test code (unless we have an instance from one of the test lines)
		return Execute(method, instance, args, false);
	}

	private readonly HashSet<Method> validatedMethods = [];

	private ValueInstance Execute(Method method, ValueInstance? instance,
		IReadOnlyList<ValueInstance> args, bool runInlineTests)
	{
		if (args.Count > method.Parameters.Count)
			throw new TooManyArguments(args[method.Parameters.Count].ToString(), args, method);
		var context = new ExecutionContext(method.Type) { This = instance };
		for (var i = 0; i < method.Parameters.Count; i++)
		{
			var param = method.Parameters[i];
			var arg = i < args.Count
				? args[i]
				: param.DefaultValue != null
					? RunExpression(param.DefaultValue, context)
					: throw new MissingArgument(param.Name, args, method);
			context.Set(param.Name, arg);
		}
		try
		{
			return RunExpression(method.GetBodyAndParseIfNeeded(), context, runInlineTests);
		}
		catch (ReturnSignal ret)
		{
			return ret.Value;
		}
	}

	public class TooManyArguments(string argument, IEnumerable<ValueInstance> args, Method method)
		: Exception(argument + ", given arguments: " + args.ToWordList() + ", method " +
			method.Name + " requires these parameters: " + method.Parameters.ToWordList());

	public class MissingArgument(string paramName, IEnumerable<ValueInstance> args, Method method)
		: Exception(paramName + ", given arguments: " + args.ToWordList() + ", method " +
			method.Name + " requires these parameters: " + method.Parameters.ToWordList());

	public ValueInstance RunExpression(Expression expr, ExecutionContext context,
		bool runInlineTests = false) =>
		expr switch
		{
			Body body => EvaluateBody(body, context, runInlineTests),
			Value v => new ValueInstance(v.ReturnType, v.Data),
			ParameterCall or VariableCall => context.Get(expr.ToString()),
			MemberCall m => EvaluateMember(m, context),
			If iff => EvaluateIf(iff, context),
			Return r => EvaluateReturn(r, context),
			MethodCall call => EvaluateMethodCall(call, context),
			Declaration c => EvaluateAndAssign(c.Name, c.Value, context),
			MutableReassignment a => EvaluateAndAssign(a.Name, a.Value, context),
			Instance i => new ValueInstance(i.ReturnType, context.Variables.Count > 0
				? context.Variables.First().Value.Value
				: context.This != null
					? context.This.Value
					: throw new NoInstanceGiven(expr, context.Type)),
			_ => //ncrunch: no coverage start
				throw new NotSupportedException($"Expression not supported yet: {expr.GetType().Name} in {context.Type}")
			//ncrunch: no coverage end
		};

	public class NoInstanceGiven(Expression expr, Type type) : Exception(expr + " in " + type);

	private ValueInstance EvaluateBody(Body body, ExecutionContext ctx, bool runInlineTests)
	{
		ValueInstance last =
			new((ctx.This?.ReturnType.Package ?? body.Method.Type.Package).FindType(Base.None)!, null);
		if (runInlineTests)
			inlineTestDepth++;
		try
		{
			foreach (var e in body.Expressions)
			{
				var isTest = IsStandaloneInlineTest(e);
				// When validating (runInlineTests is true), skip the implementation logic lines.
				// These lines will be tested anyway when a test line calls this method.
				// When executing normally (runInlineTests is false), skip the test lines.
				if (isTest == !runInlineTests)
					continue;
				last = RunExpression(e, ctx);
				if (runInlineTests && isTest && !ToBool(last))
					throw new TestFailed(body.Method, e, last);
			}
			return last;
		}
		finally
		{
			if (runInlineTests)
				inlineTestDepth--;
		}
	}

	public sealed class TestFailed(Method method, Expression expression, ValueInstance result)
		: Exception($"\"{method.Name}\" method failed: {expression}, result: {result} in" +
			Environment.NewLine + $"{method.Type.FilePath}:line {expression.LineNumber + 1}");

	/// <summary>
	/// Evaluate inline tests at top-level only (outermost call), avoid recursion
	/// </summary>
	private int inlineTestDepth;

	/// <summary>
	/// An inline test is typically a standalone boolean expression at the method top level,
	/// not part of a control flow or assignment expression.
	/// </summary>
	private static bool IsStandaloneInlineTest(Expression e) =>
		e.ReturnType.Name == Base.Boolean &&
		e is not If &&
		e is not Return &&
		e is not Declaration &&
		e is not MutableReassignment;

	private ValueInstance EvaluateAndAssign(string name, Expression value, ExecutionContext ctx) =>
		ctx.Set(name, RunExpression(value, ctx));

	private ValueInstance EvaluateReturn(Return r, ExecutionContext ctx) =>
		throw new ReturnSignal(RunExpression(r.Value, ctx));

	private ValueInstance EvaluateIf(If iff, ExecutionContext ctx) =>
		ToBool(RunExpression(iff.Condition, ctx))
			? RunExpression(iff.Then, ctx)
			: iff.OptionalElse != null
				? RunExpression(iff.OptionalElse, ctx)
				: new ValueInstance(ctx.This?.ReturnType ?? iff.ReturnType, null);

	private ValueInstance EvaluateMember(MemberCall member, ExecutionContext ctx) =>
		member.Instance == null
			? ctx.Get(member.Member.Name)
			: member.Member.InitialValue != null
				? RunExpression(member.Member.InitialValue, ctx)
				: ctx.Get(member.Member.Name);

	private ValueInstance EvaluateMethodCall(MethodCall call, ExecutionContext ctx)
	{
		var op = call.Method.Name;
		if (IsArithmetic(op) || IsCompare(op))
			return EvaluateArithmeticOrCompare(call, ctx);
		// If the call has an instance (like 'not true'), evaluate it.
		// Otherwise, if we are already inside an instance method, use 'ctx.This'.
		var instance = call.Instance != null
			? RunExpression(call.Instance, ctx)
			: ctx.This;
		var args = new List<ValueInstance>(call.Arguments.Count);
		foreach (var a in call.Arguments)
			args.Add(RunExpression(a, ctx));
		return Execute(call.Method, instance, args);
	}

	private static bool IsArithmetic(string name) =>
		name is BinaryOperator.Plus or BinaryOperator.Minus or BinaryOperator.Multiply
			or BinaryOperator.Divide or BinaryOperator.Modulate;

	private static bool IsCompare(string name) =>
		name is BinaryOperator.Greater or BinaryOperator.Smaller or BinaryOperator.Is;

	private ValueInstance EvaluateArithmeticOrCompare(MethodCall call, ExecutionContext ctx)
	{
		if (call.Instance == null || call.Arguments.Count != 1)
			throw new InvalidOperationException("Binary call must have instance and 1 argument"); //ncrunch: no coverage
		var left = RunExpression(call.Instance, ctx).Value;
		var right = RunExpression(call.Arguments[0], ctx).Value;
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
				_ => throw new NotSupportedException("Operator not supported for Number: " + op) //ncrunch: no coverage
			};
		}
		return op switch
		{
			BinaryOperator.Greater => Bool(NumberToDouble(left) > NumberToDouble(right)),
			BinaryOperator.Smaller => Bool(NumberToDouble(left) < NumberToDouble(right)),
			BinaryOperator.Is => Bool(Equals(left, right)),
			_ => throw new NotSupportedException("Operator not supported for Boolean: " + op) //ncrunch: no coverage
		};
	}

	private static bool ToBool(ValueInstance v) =>
		v.Value switch
		{
			bool b => b,
			Value { ReturnType.Name: Base.Boolean, Data: bool bv } => bv, //ncrunch: no coverage
			_ => throw new InvalidOperationException("Expected Boolean, got: " + v) //ncrunch: no coverage
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