using System.Globalization;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public sealed class Executor(Package basePackage, TestBehavior behavior = TestBehavior.OnFirstRun)
{
	public ValueInstance Execute(Method method, ValueInstance? instance,
		IReadOnlyList<ValueInstance> args)
	{
		ValueInstance? returnValue = null;
		if (inlineTestDepth == 0 && behavior != TestBehavior.Disabled && validatedMethods.Add(method))
			returnValue = Execute(method, instance, args, true);
		if (inlineTestDepth > 0 || behavior != TestBehavior.TestRunner)
			returnValue = Execute(method, instance, args, false);
		return returnValue ?? new ValueInstance(method.ReturnType, null);
	}

	/// <summary>
	/// Evaluate inline tests at top-level only (outermost call), avoid recursion
	/// </summary>
	private int inlineTestDepth;
	private readonly HashSet<Method> validatedMethods = [];

	private ValueInstance Execute(Method method, ValueInstance? instance,
		IReadOnlyList<ValueInstance> args, bool runOnlyTests)
	{
		if (args.Count > method.Parameters.Count)
			throw new TooManyArguments(args[method.Parameters.Count].ToString(), args, method);
		// If we are in a from constructor, create the instance here
		if (method.Name == Method.From)
		{
			if (instance != null)
				throw new MethodCall.CannotCallFromConstructorWithExistingInstance();
			instance = new ValueInstance(method.Type, args.Count > 0
				? args[0].Value
				: null);
		}
		var context = new ExecutionContext(method.Type) { This = instance };
		if (!runOnlyTests)
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
			var body = method.GetBodyAndParseIfNeeded();
			//TODO: now everything fails, but all we do is have this extra check
			if (body is not Body && runOnlyTests)
				return new ValueInstance(method.ReturnType, true);
			return RunExpression(body, context, runOnlyTests);
		}
		catch (ReturnSignal ret)
		{
			return ret.Value;
		}
		catch (TestFailed)
		{
			throw;
		}
		catch (Exception ex)
		{
			throw new MethodExecutionFailed(method, context, ex);
		}
	}

	public class TooManyArguments(string argument, IEnumerable<ValueInstance> args, Method method)
		: ParsingFailed(method.Type, method.TypeLineNumber,
			argument + ", given arguments: " + args.ToWordList() + ", method " + method.Name +
			" requires these parameters: " + method.Parameters.ToWordList());

	public class MethodExecutionFailed(Method method, ExecutionContext context, Exception inner)
		: ParsingFailed(method.Type, method.TypeLineNumber, context.ToString()!, inner);

	public class MissingArgument(string paramName, IEnumerable<ValueInstance> args, Method method)
		: ParsingFailed(method.Type, method.TypeLineNumber,
			paramName + ", given arguments: " + args.ToWordList() + ", method " + method.Name +
			" requires these parameters: " + method.Parameters.ToWordList());

	public ValueInstance RunExpression(Expression expr, ExecutionContext context,
		bool runOnlyTests = false) =>
		expr switch
		{
			Body body => EvaluateBody(body, context, runOnlyTests),
			Value v => new ValueInstance(v.ReturnType, v.Data),
			ParameterCall or VariableCall => context.Get(expr.ToString()),
			MemberCall m => EvaluateMember(m, context),
			If iff => EvaluateIf(iff, context),
			Return r => EvaluateReturn(r, context),
			MethodCall call => EvaluateMethodCall(call, context),
			Declaration c => EvaluateAndAssign(c.Name, c.Value, context),
			MutableReassignment a => EvaluateAndAssign(a.Name, a.Value, context),
			Instance i => ExtractValue(i.ReturnType, expr, context),
			_ => //ncrunch: no coverage start
				throw new NotSupportedException($"Expression not supported yet: {expr.GetType().Name} " +
					$"in {context.Type}")
			//ncrunch: no coverage end
		};

	private static ValueInstance ExtractValue(Type type, Expression expr, ExecutionContext context) =>
		new(type, context.Variables.ContainsKey(Base.ValueLowercase)
			? context.Variables[Base.ValueLowercase].Value
			: context.This != null
				? context.This.Value
				: throw new NoInstanceGiven(expr, context.Type));

	public class NoInstanceGiven(Expression expr, Type type) : Exception(expr + " in " + type);

	private ValueInstance EvaluateBody(Body body, ExecutionContext ctx, bool runOnlyTests)
	{
		ValueInstance last =
			new((ctx.This?.ReturnType.Package ?? body.Method.Type.Package).FindType(Base.None)!, null);
		if (runOnlyTests)
			inlineTestDepth++;
		try
		{
			foreach (var e in body.Expressions)
			{
				var isTest = !e.Equals(body.Expressions[^1]) && IsStandaloneInlineTest(e);
				// When validating (runInlineTests is true), skip the implementation logic lines.
				// These lines will be tested anyway when a test line calls this method.
				// When executing normally (runInlineTests is false), skip the test lines.
				if (isTest == !runOnlyTests)
					continue;
				last = RunExpression(e, ctx);
				if (runOnlyTests && isTest && !ToBool(last))
					throw new TestFailed(body.Method, e, last);
			}
			if (runOnlyTests && last.Value == null)
				throw new NoTestPresentInMethod(body.Method);
			return last;
		}
		finally
		{
			if (runOnlyTests)
				inlineTestDepth--;
		}
	}

	public class NoTestPresentInMethod(Method method) : Exception(method + " has no test lines, " +
		"which is not allowed in Strict. Every single method must start with at least one test!");

	public sealed class TestFailed(Method method, Expression expression, ValueInstance result)
		: Exception($"\"{method.Name}\" method failed: {expression}, result: {result} in" +
			Environment.NewLine + $"{method.Type.FilePath}:line {expression.LineNumber + 1}");

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
		ctx.Type.Members.Contains(member.Member) && ctx.This == null
			? throw new UnableToCallMemberWithoutInstance(member, ctx)
			: ctx.This?.Value is Dictionary<string, object?> dict &&
			dict.TryGetValue(member.Member.Name, out var value)
				? new ValueInstance(member.ReturnType, value)
				: member.Member.InitialValue != null
					? RunExpression(member.Member.InitialValue, ctx)
					: ctx.Get(member.Member.Name);

	public class UnableToCallMemberWithoutInstance(MemberCall member, ExecutionContext ctx)
		: Exception(member + ", context " + ctx);

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
			BinaryOperator.Is => Bool(Equals(left, right) || (left == null || right == null
				? throw new ComparisonsToNullAreNotAllowed(ctx.Type, left, right)
				: false)),
			_ => throw new NotSupportedException("Operator not supported for Boolean: " + op) //ncrunch: no coverage
		};
	}

	public class ComparisonsToNullAreNotAllowed(Type type, object? left, object? right)
		: ParsingFailed(type, type.LineNumber, $"{left} is {right}");

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