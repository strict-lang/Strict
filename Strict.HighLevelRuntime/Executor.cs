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
			// For test validation, check if the method is simple before attempting to parse
			// This avoids parsing errors when instance members are accessed but no instance exists
			if (runOnlyTests && IsSimpleSingleLineMethod(method))
				return new ValueInstance(method.ReturnType, true);
			Expression body;
			try
			{
				body = method.GetBodyAndParseIfNeeded();
			}
			catch (Exception inner) when (runOnlyTests)
			{
				throw new MethodRequiresTest(method,
					"Test execution failed with:\n" + method.lines.ToWordList("\n") + "\n" + inner);
			}
			if (body is not Body && runOnlyTests)
				return IsSimpleExpressionWithLessThanThreeSubExpressions(body)
					? new ValueInstance(method.ReturnType, true)
					: throw new MethodRequiresTest(method, body.ToString());
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
		catch (MethodRequiresTest)
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
		: ParsingFailed(method.Type, method.TypeLineNumber, context.ToString(), inner);

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
			ParameterCall or VariableCall => expr.ToString() == Base.ValueLowercase &&
				context.This != null
					? context.This
					: context.Get(expr.ToString()),
			MemberCall m => EvaluateMember(m, context),
			If iff => EvaluateIf(iff, context),
			Return r => EvaluateReturn(r, context),
			To t => EvaluateTo(t, context),
			Not n => EvaluateNot(n, context),
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
				throw new MethodRequiresTest(body.Method, body.ToString());
			return last;
		}
		finally
		{
			if (runOnlyTests)
				inlineTestDepth--;
		}
	}

	/// <summary>
	/// Check if a method is simple enough to not require tests by analyzing its raw text.
	/// This is done BEFORE parsing to avoid errors when no instance is available.
	/// We need to attempt to parse if the method might be simple, otherwise we can't validate it.
	/// This method returns true only for trivially simple cases that definitely don't need tests.
	/// </summary>
	private static bool IsSimpleSingleLineMethod(Method method)
	{
		if (method.lines.Count != 2)
			return false;
		var bodyLine = method.lines[1].Trim();
		// Count method calls - these usually add complexity
		var hasMethodCalls = bodyLine.Contains('(') && !bodyLine.StartsWith('(');
		if (hasMethodCalls)
			return false;
		// Count conditionals and operators
		var questionMarkCount = bodyLine.Count(c => c == '?');
		var operatorCount = bodyLine.Split(' ').Count(w => w is "and" or "or" or "not" or "is");
		// Simple cases: no conditionals and at most 1 operator (e.g., "value", "value is other")
		// Or: 1 conditional with a simple condition and simple branches (e.g., "value ? true else false")
		return questionMarkCount == 0 && operatorCount <= 1 ||
			questionMarkCount == 1 && operatorCount <= 2;
	}

	/// <summary>
	/// Simple expressions like "value is other" or "value" don't need tests as they are
	/// essentially getters or trivial delegations. Complex expressions need tests.
	/// </summary>
	private static bool IsSimpleExpressionWithLessThanThreeSubExpressions(Expression expr) =>
		CountExpressionComplexity(expr) <= MaxSimpleExpressionComplexity;

	private const int MaxSimpleExpressionComplexity = 3;

	private static int CountExpressionComplexity(Expression expr) =>
		expr switch
		{
			Binary => 1,
			Not n => 1 + CountExpressionComplexity(n.Instance!),
			MethodCall m => 1 + (m.Instance != null
				? CountExpressionComplexity(m.Instance)
				: 0) + m.Arguments.Sum(CountExpressionComplexity),
			If i => CountExpressionComplexity(i.Condition) + CountExpressionComplexity(i.Then) +
				(i.OptionalElse != null
					? CountExpressionComplexity(i.OptionalElse)
					: 0),
			_ => 1
		};

	public class MethodRequiresTest(Method method, string body) : ParsingFailed(method.Type,
		method.TypeLineNumber, $"Method {method.Parent.FullName}.{method.Name}\n{body}")
	{
		public MethodRequiresTest(Method method, Body body) : this(method, body+
			$" ({{CountExpressionComplexity(body)}} expressions)") { }
	}

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

	private ValueInstance EvaluateTo(To to, ExecutionContext ctx)
	{
		var left = RunExpression(to.Instance!, ctx).Value;
		if (to.ConversionType.Name == Base.Text)
			return new ValueInstance(to.ConversionType, left?.ToString() ?? "");
		if (to.ConversionType.Name == Base.Number)
			return new ValueInstance(to.ConversionType, NumberToDouble(left));
		throw new NotSupportedException("Conversion to " + to.ConversionType.Name + " not supported");
	}

	private ValueInstance EvaluateNot(Not not, ExecutionContext ctx) =>
		Bool(!ToBool(RunExpression(not.Instance!, ctx)));

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
		if (IsArithmetic(op) || IsCompare(op) || IsLogical(op))
			return EvaluateArithmeticOrCompareOrLogical(call, ctx);
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
			or BinaryOperator.Divide or BinaryOperator.Modulate or BinaryOperator.Power;

	private static bool IsCompare(string name) =>
		name is BinaryOperator.Greater or BinaryOperator.Smaller or BinaryOperator.Is
			or BinaryOperator.GreaterOrEqual or BinaryOperator.SmallerOrEqual or BinaryOperator.IsNot;

	private static bool IsLogical(string name) =>
		name is BinaryOperator.And or BinaryOperator.Or or BinaryOperator.Xor or UnaryOperator.Not;

	private ValueInstance EvaluateArithmeticOrCompareOrLogical(MethodCall call, ExecutionContext ctx)
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
				BinaryOperator.Power => Number(Math.Pow(l, r)),
				_ => throw new NotSupportedException("Operator not supported for Number: " + op) //ncrunch: no coverage
			};
		}
		if (IsCompare(op))
		{
			if (op is BinaryOperator.Is or BinaryOperator.IsNot)
				return op is BinaryOperator.Is
					? Bool(Equals(left, right) || (left == null || right == null
						? throw new ComparisonsToNullAreNotAllowed(ctx.Type, left, right)
						: false))
					: Bool(!Equals(left, right) || (left == null || right == null
						? throw new ComparisonsToNullAreNotAllowed(ctx.Type, left, right)
						: false));
			var l = NumberToDouble(left);
			var r = NumberToDouble(right);
			return op switch
			{
				BinaryOperator.Greater => Bool(l > r),
				BinaryOperator.Smaller => Bool(l < r),
				BinaryOperator.GreaterOrEqual => Bool(l >= r),
				BinaryOperator.SmallerOrEqual => Bool(l <= r),
				_ => throw new NotSupportedException("Operator not supported for Compare: " + op) //ncrunch: no coverage
			};
		}
		return op switch
		{
			BinaryOperator.And => Bool(ToBool(left) && ToBool(right)),
			BinaryOperator.Or => Bool(ToBool(left) || ToBool(right)),
			BinaryOperator.Xor => Bool(ToBool(left) ^ ToBool(right)),
			_ => throw new NotSupportedException("Operator not supported for Logical: " + op) //ncrunch: no coverage
		};
	}

	private ValueInstance Bool(bool b) => new(boolType, b);
	private readonly Type boolType = basePackage.FindType(Base.Boolean)!;

	private static bool ToBool(object? v) =>
		v switch
		{
			bool b => b,
			ValueInstance vi => ToBool(vi.Value),
			Value { ReturnType.Name: Base.Boolean, Data: bool bv } => bv, //ncrunch: no coverage
			_ => throw new InvalidOperationException("Expected Boolean, got: " + v) //ncrunch: no coverage
		};

	public class ComparisonsToNullAreNotAllowed(Type type, object? left, object? right)
		: ParsingFailed(type, type.LineNumber, $"{left} is {right}");

	private ValueInstance Number(double n) => new(numberType, n);
	private readonly Type numberType = basePackage.FindType(Base.Number)!;

	private static double NumberToDouble(object? n) =>
		Convert.ToDouble(n, CultureInfo.InvariantCulture);

	private sealed class ReturnSignal(ValueInstance value) : Exception
	{
		public ValueInstance Value { get; } = value;
	}
}