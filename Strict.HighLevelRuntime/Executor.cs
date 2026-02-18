using Strict.Expressions;
using Strict.Language;
using System.Collections;
using System.Collections.Generic;
using static Strict.HighLevelRuntime.ExecutionContext;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public sealed class Executor(Package basePackage, TestBehavior behavior = TestBehavior.OnFirstRun)
{
	public ValueInstance Execute(Method method, ValueInstance? instance,
		IReadOnlyList<ValueInstance> args, ExecutionContext? parentContext = null)
	{
		ValueInstance? returnValue = null;
    if (inlineTestDepth == 0 && behavior != TestBehavior.Disabled &&
			(behavior == TestBehavior.TestRunner || validatedMethods.Add(method)))
			returnValue = Execute(method, instance, args, parentContext, true);
		if (inlineTestDepth > 0 || behavior != TestBehavior.TestRunner)
			returnValue = Execute(method, instance, args, parentContext, false);
		return returnValue ?? new ValueInstance(method.ReturnType, null);
	}

	/// <summary>
	/// Evaluate inline tests at top-level only (outermost call), avoid recursion
	/// </summary>
	private int inlineTestDepth;
	private readonly HashSet<Method> validatedMethods = [];
	private BodyEvaluator? bodyEvaluator;
	private IfEvaluator? ifEvaluator;
	private ForEvaluator? forEvaluator;
	private MethodCallEvaluator? methodCallEvaluator;
	private ListCallEvaluator? listCallEvaluator;
	private MemberCallEvaluator? memberCallEvaluator;
	private ToEvaluator? toEvaluator;
	private BodyEvaluator BodyEvaluator => bodyEvaluator ??= new BodyEvaluator(this);
	private IfEvaluator IfEvaluator => ifEvaluator ??= new IfEvaluator(this);
	private ForEvaluator ForEvaluator => forEvaluator ??= new ForEvaluator(this);
	private MethodCallEvaluator MethodCallEvaluator =>
		methodCallEvaluator ??= new MethodCallEvaluator(this);
	private ListCallEvaluator ListCallEvaluator => listCallEvaluator ??= new ListCallEvaluator(this);
	private MemberCallEvaluator MemberCallEvaluator =>
		memberCallEvaluator ??= new MemberCallEvaluator(this);
	private ToEvaluator ToEvaluator => toEvaluator ??= new ToEvaluator(this);

	private ValueInstance Execute(Method method, ValueInstance? instance,
		IReadOnlyList<ValueInstance> args, ExecutionContext? parentContext, bool runOnlyTests)
	{
		if (instance != null && !instance.ReturnType.IsSameOrCanBeUsedAs(method.Type))
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
		// If we are in a from constructor, create the instance here (but only for concrete types)
		if (method.Name == Method.From && !method.Type.IsGeneric)
		{
			if (instance != null)
				throw new MethodCall.CannotCallFromConstructorWithExistingInstance();
			instance = new ValueInstance(method.Type, GetFromConstructorValue(method, args));
			if (method.lines.Last() == "\tvalue" || IsTestOnlyFromMethod(method))
				return instance;
		}
		if (parentContext != null && parentContext.Method == method &&
			(parentContext.This?.Equals(instance) ?? instance == null) &&
			DoArgumentsMatch(method, args, parentContext.Variables))
			throw new StackOverflowCallingItselfWithSameInstanceAndArguments(method, instance, args,
				parentContext);
		var context =
			new ExecutionContext(method.Type, method) { This = instance, Parent = parentContext };
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
		AddDictionaryElementsAlias(context, instance);
		try
		{
			// For test validation, check if the method is simple before attempting to parse
			// This avoids parsing errors when instance members are accessed but no instance exists
			if (runOnlyTests && IsSimpleSingleLineMethod(method))
				return Bool(true);
			Expression body;
			try
			{
				body = method.GetBodyAndParseIfNeeded(runOnlyTests && method.Type.IsGeneric);
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
			var result = RunExpression(body, context, runOnlyTests);
			return !runOnlyTests && !method.ReturnType.IsMutable && result.ReturnType.IsMutable &&
				((GenericTypeImplementation)result.ReturnType).ImplementationTypes[0].
				IsSameOrCanBeUsedAs(method.ReturnType)
					? new ValueInstance(method.ReturnType, result.Value)
					: result;
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

	private static bool DoArgumentsMatch(Method method, IReadOnlyList<ValueInstance> args,
		IReadOnlyDictionary<string, ValueInstance> parentContextVariables)
	{
		for (var index = 0; index < args.Count; index++)
			if (!parentContextVariables.TryGetValue(method.Parameters[index].Name,
					out var previousArgumentValue) || !args[index].Equals(previousArgumentValue))
				return false;
		return true;
	}

	public sealed class StackOverflowCallingItselfWithSameInstanceAndArguments(
		Method method,
		ValueInstance? instance,
		IEnumerable<ValueInstance> args,
		ExecutionContext parentContext) : ExecutionFailed(method,
		"Parent context=" + parentContext + ", Instance=" + instance + ", arguments=" +
		args.ToWordList());

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
				!parameter.Type.IsIterator && !IsSingleCharacterTextArgument(parameter.Type, args[index]))
				throw new InvalidTypeForArgument(type, args, index);
			var memberName = type.Members.FirstOrDefault(member =>
					member.Name.Equals(parameter.Name, StringComparison.OrdinalIgnoreCase))?.Name ??
				GetMemberName(parameter.Name);
			result.Add(memberName, TryConvertSingleCharacterText(parameter.Type, args[index]));
		}
		return result;
	}

	private static object? GetFromConstructorValue(Method method, IReadOnlyList<ValueInstance> args)
	{
		if (args.Count == 0)
			return method.Type.Name == Base.Text
				? ""
				: TryCreateEmptyCollectionForType(method.Type);
     if (method.Type is GenericTypeImplementation { Generic.Name: Base.Dictionary } &&
			args.Count == 1 && args[0].ReturnType.IsIterator)
		{
     return args[0].Value is IDictionary dictionary
				? dictionary
				: BuildDictionaryFromPairs(args[0].Value as IList ?? Array.Empty<object?>());
		}
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

	private static object? TryCreateEmptyCollectionForType(Type type)
	{
		if (type is GenericTypeImplementation { Generic.Name: Base.Dictionary })
		{
      return new Dictionary<object, object?>();
		}
		return null;
	}

	private static void AddDictionaryElementsAlias(ExecutionContext context,
		ValueInstance? instance)
	{
		if (instance?.ReturnType is not GenericTypeImplementation { Generic.Name: Base.Dictionary })
			return;
    if (instance.Value is not IDictionary dictionaryValues ||
			context.Variables.ContainsKey(Type.ElementsLowercase))
			return;
   var listMemberType = instance.ReturnType.Members.FirstOrDefault(member =>
			member.Type is GenericTypeImplementation { Generic.Name: Base.List } ||
			member.Type.Name == Base.List)?.Type ?? instance.ReturnType.GetType(Base.List);
		var listValue = ExecutionContext.BuildDictionaryPairsList(dictionaryValues, listMemberType,
			((GenericTypeImplementation)instance.ReturnType).ImplementationTypes[0],
			((GenericTypeImplementation)instance.ReturnType).ImplementationTypes[1]);
		context.Set(Type.ElementsLowercase,
			new ValueInstance(listMemberType, listValue));
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

	public sealed class TooManyArgumentsForTypeMembersInFromConstructor(
		Type type,
		IEnumerable<ValueInstance> args) : ExecutionFailed(type,
		"args=" + args.ToWordList() + ", type=" + type + " Members=" + type.Members.ToWordList());

	public sealed class InvalidTypeForArgument(
		Type type,
		IReadOnlyList<ValueInstance> args,
		int index) : ExecutionFailed(type,
		args[index] + " at index=" + index + " does not match type=" + type + " Member=" +
		type.Members[index]);

	public sealed class CannotCallMethodWithWrongInstance(Method method, ValueInstance instance)
		: ExecutionFailed(method, instance.ToString());

	public sealed class
		TooManyArguments(Method method, string argument, IEnumerable<ValueInstance> args)
		: ExecutionFailed(method,
			argument + ", given arguments: " + args.ToWordList() + ", method " + method.Name +
			" requires these parameters: " + method.Parameters.ToWordList());

	public sealed class ArgumentDoesNotMapToMethodParameters(Method method, string message)
		: ExecutionFailed(method, message);

	public sealed class MethodExecutionFailed(
		Method method,
		ExecutionContext context,
		Exception inner) : ExecutionFailed(method, context.ToString(), inner);

	public sealed class
		MissingArgument(Method method, string paramName, IEnumerable<ValueInstance> args)
		: ExecutionFailed(method,
			paramName + ", given arguments: " + args.ToWordList() + ", method " + method.Name +
			" requires these parameters: " + method.Parameters.ToWordList());

	public ValueInstance RunExpression(Expression expr, ExecutionContext context,
		bool runOnlyTests = false) =>
		expr switch
		{
			//TODO: this class has grown too big, separate out these different evaluators into own classes
     Body body => BodyEvaluator.Evaluate(body, context, runOnlyTests),
     Value v => v.Data as ValueInstance ??
				CreateValueInstance(v.ReturnType, v.Data, context),
			ParameterCall or VariableCall => EvaluateVariable(expr.ToString(), context),
     MemberCall m => MemberCallEvaluator.Evaluate(m, context),
			ListCall listCall => ListCallEvaluator.Evaluate(listCall, context),
			If iff => IfEvaluator.Evaluate(iff, context),
			For f => ForEvaluator.Evaluate(f, context),
			Return r => EvaluateReturn(r, context),
     To t => ToEvaluator.Evaluate(t, context),
			Not n => EvaluateNot(n, context),
     MethodCall call => EvaluateMethodCall(call, context),
			Declaration c => EvaluateAndAssign(c.Name, c.Value, context),
			MutableReassignment a => EvaluateAndAssign(a.Name, a.Value, context),
			Instance => EvaluateVariable(Type.ValueLowercase, context),
			_ => throw new ExpressionNotSupported(expr, context) //ncrunch: no coverage
		};

	private object EvaluateValueData(object valueData, ExecutionContext context) =>
		valueData switch
		{
			Expression valueExpression => RunExpression(valueExpression, context),
			IList<Expression> list => list.Select(e => RunExpression(e, context)).ToList(),
			_ => valueData
		};

	private ValueInstance CreateValueInstance(Type returnType, object valueData,
		ExecutionContext context)
	{
		var value = EvaluateValueData(valueData, context);
		if (IsDictionaryType(returnType))
			value = NormalizeDictionaryValue(returnType, value);
		return new ValueInstance(returnType, value);
	}

	private static bool IsDictionaryType(Type type) =>
		type.Name == Base.Dictionary ||
		type is GenericTypeImplementation { Generic.Name: Base.Dictionary };

	private static object NormalizeDictionaryValue(Type dictionaryType, object value)
	{
		if (value is IDictionary<string, object?> rawMembers)
		{
			var listMemberName = dictionaryType.Members.FirstOrDefault(member =>
				member.Type is GenericTypeImplementation { Generic.Name: Base.List } ||
				member.Type.Name == Base.List)?.Name;
			if (listMemberName != null && rawMembers.TryGetValue(listMemberName, out var listObj))
        return BuildDictionaryFromPairs(listObj as IList ?? Array.Empty<object?>());
		}
		if (value is IList list)
			return BuildDictionaryFromPairs(list);
		return value;
	}

  private static IDictionary BuildDictionaryFromPairs(IList pairs)
	{
    var dictionary = new Dictionary<object, object?>();
		foreach (var pair in pairs)
     if (TryGetPairValues(pair, out var key, out var pairValue) && key != null)
				dictionary[key] = pairValue;
		return dictionary;
	}

	private static bool TryGetPairValues(object? pair, out object? key, out object? value)
	{
		key = null;
		value = null;
		var pairList = pair switch
		{
			ValueInstance { Value: IList list } => list,
			IList list => list,
			_ => null
		};
		if (pairList == null || pairList.Count < 2)
			return false;
		key = ExtractRawValue(pairList[0]);
		value = ExtractRawValue(pairList[1]);
		return true;
	}

	private static object? ExtractRawValue(object? value) =>
		value is ValueInstance instance
			? instance.Value
			: value;

	public class ExpressionNotSupported(Expression expr, ExecutionContext context)
		: ExecutionFailed(context.Type, expr.GetType().Name);

	private static ValueInstance EvaluateVariable(string name, ExecutionContext context) =>
		context.Find(name) ?? (name == Type.ValueLowercase
			? context.This
			: null) ?? throw new VariableNotFound(name, context.Type, context.This);

	internal void IncrementInlineTestDepth() => inlineTestDepth++;
	internal void DecrementInlineTestDepth() => inlineTestDepth--;
	internal ValueInstance EvaluateMethodCall(MethodCall call, ExecutionContext ctx) =>
		MethodCallEvaluator.Evaluate(call, ctx);

	public sealed class ReturnTypeMustMatchMethod(Body body, ValueInstance last) : ExecutionFailed(
		body.Method,
		"Return value " + last + " does not match method " + body.Method.Name + " ReturnType=" +
		body.Method.ReturnType);

	/// <summary>
	/// A from method is test-only when all body lines are test assertions and there is no
	/// implementation line. Such from methods create a default instance without any computation.
	/// </summary>
	private static bool IsTestOnlyFromMethod(Method method) =>
		method.lines.Count > 1 && Enumerable.Range(1, method.lines.Count - 1).All(i =>
			method.lines[i].Contains(" is ", StringComparison.Ordinal) &&
			!method.lines[i].Contains("?"));

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

	public sealed class TestFailed(
		Method method,
		Expression expression,
		ValueInstance result,
		string details) : ExecutionFailed(method,
		$"\"{method.Name}\" method failed: {expression}, result: {result}" + (details.Length > 0
			? $", evaluated: {details}"
			: "") + " in" + Environment.NewLine +
		$"{method.Type.FilePath}:line {expression.LineNumber + 1}");

	private ValueInstance EvaluateAndAssign(string name, Expression value, ExecutionContext ctx) =>
		ctx.Set(name, RunExpression(value, ctx));

	private ValueInstance EvaluateReturn(Return r, ExecutionContext ctx) =>
		throw new ReturnSignal(RunExpression(r.Value, ctx));

	private ValueInstance EvaluateNot(Not not, ExecutionContext ctx) =>
		Bool(!ToBool(RunExpression(not.Instance!, ctx)));

	public class UnableToCallMemberWithoutInstance(MemberCall member, ExecutionContext ctx)
		: Exception(member + ", context " + ctx);

  internal ValueInstance None() => new(noneType, null);
	private readonly Type noneType = basePackage.FindType(Base.None)!;
 internal ValueInstance Bool(bool b) => new(boolType, b);
	private readonly Type boolType = basePackage.FindType(Base.Boolean)!;
	private readonly Type genericListType = basePackage.FindType(Base.List)!;
	private readonly Type errorType = basePackage.FindType(Base.Error)!;
	private readonly Type stacktraceType = basePackage.FindType(Base.Stacktrace)!;
	private readonly Type methodType = basePackage.FindType(Base.Method)!;
	private readonly Type typeType = basePackage.FindType(Base.Type)!;
	public const string ListsHaveDifferentDimensions = "listsHaveDifferentDimensions";
	internal Type BoolType => boolType;
	internal Type NumberType => numberType;
	internal Type GenericListType => genericListType;
	internal Type ErrorType => errorType;
	internal Type StacktraceType => stacktraceType;
	internal Type MethodType => methodType;
	internal Type TypeType => typeType;

  internal static bool ToBool(object? v) =>
		v switch
		{
			bool b => b,
			ValueInstance vi => ToBool(vi.Value),
			Value { ReturnType.Name: Base.Boolean, Data: bool bv } => bv,
			_ => throw new InvalidOperationException("Expected Boolean, got: " + v)
		};

	public sealed class ComparisonsToNullAreNotAllowed(Method method, object? left, object? right)
		: ExecutionFailed(method, $"{left} is {right}");

 internal ValueInstance Number(double n) => new(numberType, n);
	private readonly Type numberType = basePackage.FindType(Base.Number)!;

	private sealed class ReturnSignal(ValueInstance value) : Exception
	{
		public ValueInstance Value { get; } = value;
	}
}