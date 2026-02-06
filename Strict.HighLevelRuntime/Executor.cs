using System.Collections;
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
		if (instance != null && instance.ReturnType != method.Parent)
			throw new CannotCallMethodWithWrongInstance(method, instance);
		if (args.Count > method.Parameters.Count)
			throw new TooManyArguments(method, args[method.Parameters.Count].ToString(), args);
		for (var index = 0; index < args.Count; index++)
			if (!args[index].ReturnType.IsSameOrCanBeUsedAs(method.Parameters[index].Type) &&
				!method.Parameters[index].Type.IsIterator &&
				!IsSingleCharacterTextArgument(method.Parameters[index].Type, args[index]))
				throw new ArgumentDoesNotMapToMethodParameters(method,
					"Method \"" + method + "\" parameter " + index + ": " +
					method.Parameters[index].ToStringWithInnerMembers() +
					" cannot be assigned from argument " + args[index] + " " + args[index].ReturnType);
		// If we are in a from constructor, create the instance here
		if (method.Name == Method.From)
		{
			if (instance != null)
				throw new MethodCall.CannotCallFromConstructorWithExistingInstance();
			instance = new ValueInstance(method.Type, GetFromConstructorValue(method, args));
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
						: throw new MissingArgument(method, param.Name, args);
				context.Set(param.Name, arg);
			}
		try
		{
			// For test validation, check if the method is simple before attempting to parse
			// This avoids parsing errors when instance members are accessed but no instance exists
			if (runOnlyTests && IsSimpleSingleLineMethod(method))
				return Bool(true);
			Expression body;
			try
			{
				body = method.GetBodyAndParseIfNeeded();
			}
			catch (Exception inner) when (runOnlyTests)
			{
				throw new MethodRequiresTest(method,
					$"Test execution failed: {method.Parent.FullName}.{method.Name}\n" +
					method.lines.ToWordList("\n") + "\n" + inner);
			}
			if (body is not Body && runOnlyTests)
				return IsSimpleExpressionWithLessThanThreeSubExpressions(body)
					? Bool(true)
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
		catch (ExecutionFailed)
		{
			throw;
		}
		catch (Exception ex)
		{
			throw new MethodExecutionFailed(method, context, ex);
		}
	}

	private static IDictionary<string, object?> ConvertFromArgumentsToDictionary(Method fromMethod,
		IReadOnlyList<ValueInstance> args)
	{
		var type = fromMethod.Type;
		if (args.Count > fromMethod.Parameters.Count)
			throw new TooManyArgumentsForTypeMembersInFromConstructor(type, args);
		var result = new Dictionary<string, object?>(StringComparer.Ordinal);
		for (var index = 0; index < args.Count; index++)
		{
			var parameter = fromMethod.Parameters[index];
			if (!args[index].ReturnType.IsSameOrCanBeUsedAs(parameter.Type) &&
				!parameter.Type.IsIterator &&
				!IsSingleCharacterTextArgument(parameter.Type, args[index]))
				throw new InvalidTypeForArgument(type, args, index);
			result.Add(GetMemberName(parameter.Name),
				TryConvertSingleCharacterText(parameter.Type, args[index]));
		}
		return result;
	}

	private static object? GetFromConstructorValue(Method method, IReadOnlyList<ValueInstance> args)
	{
		if (args.Count == 0)
			return null;
		if (args.Count == 1)
		{
			var arg = args[0];
			if (method.Type.Name == Base.Character && IsSingleCharacterTextArgument(method.Type, arg))
				return (int)((string)arg.Value!)[0];
			if (method.Type.IsSameOrCanBeUsedAs(arg.ReturnType) &&
				!IsSingleCharacterTextArgument(method.Type, arg))
				return arg.Value;
		}
		return ConvertFromArgumentsToDictionary(method, args);
	}

	private static object? TryConvertSingleCharacterText(Type targetType, ValueInstance value)
	{
		if (value.ReturnType.Name == Base.Text && value.Value is string text && text.Length == 1 &&
			targetType.Name is Base.Number or Base.Character)
			return (int)text[0];
		return value.Value;
	}

	private static bool IsSingleCharacterTextArgument(Type targetType, ValueInstance value) =>
		value.ReturnType.Name == Base.Text && value.Value is string text && text.Length == 1 &&
		targetType.Name is Base.Number or Base.Character;

	private static string GetMemberName(string parameterName) =>
		parameterName.Length > 0
			? char.ToUpperInvariant(parameterName[0]) + parameterName[1..]
			: parameterName;

	public sealed class TooManyArgumentsForTypeMembersInFromConstructor(Type type,
		IEnumerable<ValueInstance> args) : ExecutionFailed(type,
		"args=" + args.ToWordList() + ", type=" + type + " Members=" + type.Members.ToWordList());

	public sealed class InvalidTypeForArgument(Type type, IReadOnlyList<ValueInstance> args,
		int index) : ExecutionFailed(type, args[index] + " at index=" + index +
		" does not match type=" + type + " Member=" + type.Members[index]);

	public sealed class CannotCallMethodWithWrongInstance(Method method, ValueInstance instance)
		: ExecutionFailed(method, instance.ToString());

	public sealed class TooManyArguments(Method method, string argument, IEnumerable<ValueInstance> args)
		: ExecutionFailed(method,
			argument + ", given arguments: " + args.ToWordList() + ", method " + method.Name +
			" requires these parameters: " + method.Parameters.ToWordList());

	public sealed class ArgumentDoesNotMapToMethodParameters(Method method, string message)
		: ExecutionFailed(method, message);

	public sealed class MethodExecutionFailed(Method method, ExecutionContext context, Exception inner)
		: ExecutionFailed(method, context.ToString(), inner);

	public sealed class MissingArgument(Method method, string paramName, IEnumerable<ValueInstance> args)
		: ExecutionFailed(method,
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
			For f => EvaluateFor(f, context),
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

	public sealed class NoInstanceGiven(Expression expr, Type type) : Exception(expr + " in " + type);

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
				if (isTest == !runOnlyTests && e is not Declaration && e is not MutableReassignment ||
					runOnlyTests && e is Declaration &&
					body.Method.Type.Members.Any(m => !m.IsConstant && e.ToString().Contains(m.Name)))
					continue;
				last = RunExpression(e, ctx);
				if (runOnlyTests && isTest && !ToBool(last))
					//TODO: we need to show the stacktrace from an Error inside Strict!
					throw new TestFailed(body.Method, e, last, GetTestFailureDetails(e, ctx));
			}
			if (runOnlyTests && last.Value == null && body.Method.Name != Base.Run)
				throw new MethodRequiresTest(body.Method, body);
			return last;
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

	public class MethodRequiresTest(Method method, string body) : ExecutionFailed(method,
		body.StartsWith("Test execution failed", StringComparison.Ordinal)
			? body
			: $"Method {method.Parent.FullName}.{method.Name}\n{body}")
	{
		public MethodRequiresTest(Method method, Body body) : this(method,
			body + " ({CountExpressionComplexity(body)} expressions)") { }
	}

	public sealed class TestFailed(Method method, Expression expression, ValueInstance result,
		string details)
		: ExecutionFailed(method, $"\"{method.Name}\" method failed: {expression}, result: {result}" +
			(details.Length > 0 ? $", evaluated: {details}" : "") + " in" +
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

	private string GetTestFailureDetails(Expression expression, ExecutionContext ctx)
	{
		if (expression is Binary { Method.Name: BinaryOperator.Is, Instance: not null } binary &&
			binary.Arguments.Count == 1)
			return GetBinaryComparisonDetails(binary, ctx, BinaryOperator.Is);
		if (expression is Not { Instance: Binary { Method.Name: BinaryOperator.Is } notBinary } &&
			notBinary.Arguments.Count == 1)
			return GetBinaryComparisonDetails(notBinary, ctx, "is not");
		return string.Empty;
	}

	private string GetBinaryComparisonDetails(Binary binary, ExecutionContext ctx, string op)
	{
		var left = RunExpression(binary.Instance!, ctx);
		var right = RunExpression(binary.Arguments[0], ctx);
		return $"{left} {op} {right}";
	}

	private ValueInstance EvaluateAndAssign(string name, Expression value, ExecutionContext ctx) =>
		ctx.Set(name, RunExpression(value, ctx));

	private ValueInstance EvaluateReturn(Return r, ExecutionContext ctx) =>
		throw new ReturnSignal(RunExpression(r.Value, ctx));

	private ValueInstance EvaluateIf(If iff, ExecutionContext ctx) =>
		ToBool(RunExpression(iff.Condition, ctx))
			? RunExpression(iff.Then, ctx)
			: iff.OptionalElse != null
				? RunExpression(iff.OptionalElse, ctx)
				: Bool(true);

	private ValueInstance EvaluateFor(For f, ExecutionContext ctx)
	{
		var iterator = RunExpression(f.Iterator, ctx);
		var values = GetForValues(iterator);
		if (iterator.ReturnType.Name == Base.Range)
			return Number(values.Sum(NumberToDouble));
		double numberResult = 0;
		var shouldSum = !IsListIterator(iterator.ReturnType);
		var results = new List<object?>(values.Count);
		for (var index = 0; index < values.Count; index++)
		{
			var loopContext = new ExecutionContext(ctx.Type) { This = ctx.This, Parent = ctx };
			loopContext.Set(Base.IndexLowercase, new ValueInstance(numberType, index));
			loopContext.Set(Base.ValueLowercase, new ValueInstance(GetForValueType(iterator), values[index]));
			foreach (var customVariable in f.CustomVariables)
				if (customVariable is VariableCall variableCall)
					loopContext.Set(variableCall.Variable.Name,
						new ValueInstance(variableCall.ReturnType, values[index]));
			var result = RunExpression(f.Body, loopContext);
			if (shouldSum && result.Value is IConvertible)
				numberResult += NumberToDouble(result.Value);
			else
			{
				shouldSum = false;
				results.Add(result.Value);
			}
		}
		if (!shouldSum && !IsListIterator(iterator.ReturnType) && TrySumResults(results, out var sum))
			return Number(sum);
		return shouldSum
			? Number(numberResult)
			: new ValueInstance(listType, results);
	}

	private static bool TrySumResults(IReadOnlyList<object?> results, out double sum)
	{
		sum = 0;
		for (var i = 0; i < results.Count; i++)
		{
			var value = results[i] is ValueInstance instance
				? instance.Value
				: results[i];
			if (value is not IConvertible)
				return false;
			sum += NumberToDouble(value);
		}
		return true;
	}

	private static IReadOnlyList<object?> GetForValues(ValueInstance iterator) =>
		iterator.Value switch
		{
			IDictionary<string, object?> dict when iterator.ReturnType.Name == Base.Range =>
				GetRangeValues(dict),
			IList list => list.Cast<object?>().ToList(),
			int count => Enumerable.Range(0, count).Cast<object?>().ToList(),
			double countDouble => Enumerable.Range(0, (int)countDouble).Cast<object?>().ToList(),
			_ => throw new NotSupportedException("Iterator not supported: " + iterator.ReturnType.Name)
		};

	private static IReadOnlyList<object?> GetRangeValues(IDictionary<string, object?> dict)
	{
		var start = Convert.ToInt32(dict["Start"]);
		var end = Convert.ToInt32(dict["ExclusiveEnd"]);
		var range = new List<object?>();
		for (var i = start; i < end; i++)
			range.Add((double)i);
		return range;
	}

	private Type GetForValueType(ValueInstance iterator) =>
		iterator.ReturnType is GenericTypeImplementation { Generic.Name: Base.List } listType
			? listType.ImplementationTypes[0]
			: numberType;

	private static bool IsListIterator(Type type) =>
		type.Name == Base.List || type is GenericTypeImplementation { Generic.Name: Base.List };

	private static bool IsValueOnlyForBody(Expression body) =>
		body is VariableCall { Variable.Name: Base.ValueLowercase } ||
		body is Body { Expressions.Count: 1 } forBody &&
			IsValueOnlyForBody(forBody.Expressions[0]);

	private static bool AreInstancesEqual(ValueInstance left, ValueInstance right)
	{
		if (!left.ReturnType.IsSameOrCanBeUsedAs(right.ReturnType) &&
			!right.ReturnType.IsSameOrCanBeUsedAs(left.ReturnType))
			return false;
		return AreValuesEqual(left.Value, right.Value);
	}

	private static bool AreValuesEqual(object? left, object? right)
	{
		if (ReferenceEquals(left, right))
			return true;
		left = left is ValueInstance leftInstance ? leftInstance.Value : left;
		right = right is ValueInstance rightInstance ? rightInstance.Value : right;
		if (left is IDictionary leftDict && right is IDictionary rightDict)
			return AreDictionariesEqual(leftDict, rightDict);
		if (left is IList leftList && right is IList rightList)
			return AreListsEqual(leftList, rightList);
		if (IsNumeric(left) && IsNumeric(right))
			return NumberToDouble(left) == NumberToDouble(right);
		return Equals(left, right);
	}

	private static bool AreListsEqual(IList left, IList right)
	{
		if (left.Count != right.Count)
			return false;
		for (var i = 0; i < left.Count; i++)
			if (!AreValuesEqual(left[i], right[i]))
				return false;
		return true;
	}

	private static bool AreDictionariesEqual(IDictionary left, IDictionary right)
	{
		if (left.Count != right.Count)
			return false;
		foreach (DictionaryEntry entry in left)
		{
			if (!right.Contains(entry.Key))
				return false;
			if (!AreValuesEqual(entry.Value, right[entry.Key]))
				return false;
		}
		return true;
	}

	private static bool IsNumeric(object? value) =>
		value is sbyte or byte or short or ushort or int or uint or long or ulong or float or double
			or decimal;

	private ValueInstance EvaluateTo(To to, ExecutionContext ctx)
	{
		var left = RunExpression(to.Instance!, ctx).Value;
		if (to.Instance!.ReturnType.Name == Base.Text && to.ConversionType.Name == Base.Number &&
			left is string textValue)
			return new ValueInstance(to.ConversionType,
				double.Parse(textValue, CultureInfo.InvariantCulture));
		if (!to.Method.IsTrait && to.Method.Type.Name != Base.Number)
			return EvaluateMethodCall(to, ctx);
		if (to.ConversionType.Name == Base.Text)
			return new ValueInstance(to.ConversionType, left?.ToString() ?? "");
		if (to.ConversionType.Name == Base.Number)
			return new ValueInstance(to.ConversionType, NumberToDouble(left));
		if (!to.Method.IsTrait)
			return EvaluateMethodCall(to, ctx);
		throw new NotSupportedException("Conversion to " + to.ConversionType.Name + " not supported"); //ncrunch: no coverage
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
			: call.Method.Name != Method.From
				? ctx.This
				: null;
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
			or BinaryOperator.GreaterOrEqual or BinaryOperator.SmallerOrEqual or UnaryOperator.Not;

	private static bool IsLogical(string name) =>
		name is BinaryOperator.And or BinaryOperator.Or or BinaryOperator.Xor or UnaryOperator.Not;

	private ValueInstance EvaluateArithmeticOrCompareOrLogical(MethodCall call, ExecutionContext ctx)
	{
		if (call.Instance == null || call.Arguments.Count != 1)
			throw new InvalidOperationException("Binary call must have instance and 1 argument"); //ncrunch: no coverage
		var leftInstance = RunExpression(call.Instance, ctx);
		var rightInstance = RunExpression(call.Arguments[0], ctx);
		var left = leftInstance.Value;
		var right = rightInstance.Value;
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
			if (op is BinaryOperator.Is or UnaryOperator.Not)
			{
				if (left == null || right == null)
					throw new ComparisonsToNullAreNotAllowed(call.Method, left, right);
				// Error comparison: allow ErrorWithValue to match Error and compare specific error types
				if (rightInstance.ReturnType.IsError)
				{
					var matches = rightInstance.ReturnType.Name == Base.Error
						? leftInstance.ReturnType.IsError
						: leftInstance.ReturnType.IsSameOrCanBeUsedAs(rightInstance.ReturnType);
					return op is BinaryOperator.Is ? Bool(matches) : Bool(!matches);
				}
				//TODO: support conversions needed for Character, maybe Number <-> Text
				if (call.Instance.ReturnType.Name == Base.Character && right is string rightText)
				{
					right = (int)rightText[0];
					rightInstance = new ValueInstance(call.Instance.ReturnType, right);
				}
				if (call.Instance.ReturnType.Name == Base.Text && right is int rightInt)
				{
					right = rightInt + "";
					rightInstance = new ValueInstance(call.Instance.ReturnType, right);
				}
				//TODO: support comparison of derived types (Name <-> Text, etc.)
				//TODO: if is operator only defines (other), both left and right must match type, if not error out here
				//TODO: think about containing types, probably should not be is supported, but can be converted and then compared (e.g. File to Text is "some.txt"
				//TODO: use is operator of type if it is defined for non Number, non Text!
				var equals = AreInstancesEqual(leftInstance, rightInstance);
				return op is BinaryOperator.Is
					? Bool(equals)
					: Bool(!equals);
			}
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
	private readonly Type listType = basePackage.FindType(Base.List)!;

	private static bool ToBool(object? v) =>
		v switch
		{
			bool b => b,
			ValueInstance vi => ToBool(vi.Value),
			Value { ReturnType.Name: Base.Boolean, Data: bool bv } => bv, //ncrunch: no coverage
			_ => throw new InvalidOperationException("Expected Boolean, got: " + v) //ncrunch: no coverage
		};

	public sealed class ComparisonsToNullAreNotAllowed(Method method, object? left, object? right)
		: ExecutionFailed(method, $"{left} is {right}");

	private ValueInstance Number(double n) => new(numberType, n);
	private readonly Type numberType = basePackage.FindType(Base.Number)!;

	private static double NumberToDouble(object? n) =>
		Convert.ToDouble(n, CultureInfo.InvariantCulture);

	private sealed class ReturnSignal(ValueInstance value) : Exception
	{
		public ValueInstance Value { get; } = value;
	}
}