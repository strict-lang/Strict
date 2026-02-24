using Strict.Expressions;
using Strict.Language;
using System.Collections;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public sealed class Executor(TestBehavior behavior = TestBehavior.OnFirstRun)
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
	private BodyEvaluator BodyEvaluator => field ??= new BodyEvaluator(this);
	private IfEvaluator IfEvaluator => field ??= new IfEvaluator(this);
	private SelectorIfEvaluator SelectorIfEvaluator => field ??= new SelectorIfEvaluator(this);
	private ForEvaluator ForEvaluator => field ??= new ForEvaluator(this);
	private MethodCallEvaluator MethodCallEvaluator => field ??= new MethodCallEvaluator(this);
	private ToEvaluator ToEvaluator => field ??= new ToEvaluator(this);

	private ValueInstance Execute(Method method, ValueInstance? instance,
		IReadOnlyList<ValueInstance> args, ExecutionContext? parentContext, bool runOnlyTests)
	{
		ValidateInstanceAndArguments(method, instance, args, parentContext);
		if (method is { Name: Method.From, Type.IsGeneric: false })
			return instance != null
				? throw new MethodCall.CannotCallFromConstructorWithExistingInstance()
				: new ValueInstance(method.Type, GetFromConstructorValue(method, args));
		var context = CreateExecutionContext(method, instance, args, parentContext, runOnlyTests);
		try
		{
			if (runOnlyTests && IsSimpleSingleLineMethod(method))
				return Bool(method.ReturnType, true);
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
					? Bool(body.ReturnType, true)
					: throw new MethodRequiresTest(method, body.ToString());
			var result = RunExpression(body, context, runOnlyTests);
			if (!runOnlyTests && method.ReturnType is GenericTypeImplementation
				{
					Generic.Name: Base.Mutable
				} mutableReturnType && !result.ReturnType.IsMutable &&
				result.ReturnType.IsSameOrCanBeUsedAs(mutableReturnType.ImplementationTypes[0]))
				return new ValueInstance(method.ReturnType, result.Value);
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
	}

	private ExecutionContext CreateExecutionContext(Method method, ValueInstance? instance,
		IReadOnlyList<ValueInstance> args, ExecutionContext? parentContext, bool runOnlyTests)
	{
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
				context.Variables[param.Name] = arg;
			}
		AddDictionaryElementsAlias(context, instance);
		return context;
	}

	private static void ValidateInstanceAndArguments(Method method, ValueInstance? instance,
		IReadOnlyList<ValueInstance> args, ExecutionContext? parentContext)
	{
		if (instance != null && !instance.ReturnType.IsSameOrCanBeUsedAs(method.Type))
			throw new CannotCallMethodWithWrongInstance(method, instance); //ncrunch: no coverage
		if (args.Count > method.Parameters.Count)
			throw new TooManyArguments(method, args[method.Parameters.Count].ToString(), args);
		for (var index = 0; index < args.Count; index++)
			if (!args[index].ReturnType.IsSameOrCanBeUsedAs(method.Parameters[index].Type) &&
				!method.Parameters[index].Type.IsIterator && method.Name != Method.From &&
				!IsSingleCharacterTextArgument(method.Parameters[index].Type, args[index]))
				throw new ArgumentDoesNotMapToMethodParameters(method,
					"Method \"" + method + "\" parameter " + index + ": " +
					method.Parameters[index].ToStringWithInnerMembers() +
					" cannot be assigned from argument " + args[index] + " " + args[index].ReturnType);
		if (parentContext != null && parentContext.Method == method &&
			(parentContext.This?.Equals(instance) ?? instance == null) &&
			DoArgumentsMatch(method, args, parentContext.Variables))
			throw new StackOverflowCallingItselfWithSameInstanceAndArguments(method, instance, args,
				parentContext);
	}

	internal static ValueInstance Bool(Context any, bool b) => new(any.GetType(Base.Boolean), b);

	private static bool DoArgumentsMatch(Method method, IReadOnlyList<ValueInstance> args,
		IReadOnlyDictionary<string, ValueInstance> parentContextVariables)
	{
		for (var index = 0; index < args.Count; index++)
			if (!parentContextVariables.TryGetValue(method.Parameters[index].Name,
					out var previousArgumentValue) || !args[index].Equals(previousArgumentValue))
				return false;
		return true;
	}

	public sealed class StackOverflowCallingItselfWithSameInstanceAndArguments(Method method,
		ValueInstance? instance, IEnumerable<ValueInstance> args, ExecutionContext parentContext)
		: ExecutionFailed(method, "Parent context=" + parentContext + ", Instance=" + instance +
			", arguments=" + args.ToWordList());

	private static IDictionary<string, object?> ConvertFromArgumentsToDictionary(Method fromMethod,
		IReadOnlyList<ValueInstance> args)
	{
		var type = fromMethod.Type;
		var result = new Dictionary<string, object?>(StringComparer.Ordinal);
		for (var index = 0; index < args.Count; index++)
		{
			var parameter = fromMethod.Parameters[index];
			if (!args[index].ReturnType.IsSameOrCanBeUsedAs(parameter.Type) &&
				!parameter.Type.IsIterator && !IsSingleCharacterTextArgument(parameter.Type, args[index]))
				throw new InvalidTypeForArgument(type, args, index);
			var memberName = type.Members.FirstOrDefault(member =>
					member.Name.Equals(parameter.Name, StringComparison.OrdinalIgnoreCase))?.Name ??
				(parameter.Name.Length > 0
					? char.ToUpperInvariant(parameter.Name[0]) + parameter.Name[1..]
					: parameter.Name);
			result.Add(memberName, TryConvertSingleCharacterText(parameter.Type, args[index]));
		}
		return result;
	}

	private static object? GetFromConstructorValue(Method method, IReadOnlyList<ValueInstance> args)
	{
		if (args.Count == 0 && method.Type.Name == Base.Text)
			return "";
		if (method.Type.IsDictionary && args.Count == 1 && args[0].ReturnType.IsIterator)
			return args[0].Value as IDictionary ?? FillDictionaryFromListKeyAndValues(args[0].Value);
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

	private static void AddDictionaryElementsAlias(ExecutionContext context, ValueInstance? instance)
	{
		if (instance?.ReturnType is not GenericTypeImplementation
			{
				Generic.Name: Base.Dictionary
			} implementation || instance.Value is not Dictionary<ValueInstance, ValueInstance> dictionary ||
			context.Variables.ContainsKey(Type.ElementsLowercase))
			return;
		var listMemberType = implementation.Members.FirstOrDefault(member =>
			member.Type is GenericTypeImplementation { Generic.Name: Base.List } ||
			member.Type.Name == Base.List)?.Type ?? implementation.GetType(Base.List);
		var listValue = ExecutionContext.BuildDictionaryPairsList(listMemberType, dictionary);
		context.Set(Type.ElementsLowercase, new ValueInstance(listMemberType, listValue));
	}

	private static object? TryConvertSingleCharacterText(Type targetType, ValueInstance value) =>
		value.ReturnType.Name == Base.Text && value.Value is string { Length: 1 } text &&
		targetType.Name is Base.Number or Base.Character
			? (int)text[0]
			: value.Value;

	private static bool IsSingleCharacterTextArgument(Type targetType, ValueInstance value) =>
		value.ReturnType.Name == Base.Text && value.Value is string text && text.Length == 1 &&
		targetType.Name is Base.Number or Base.Character;

	public sealed class InvalidTypeForArgument(
		Type type,
		IReadOnlyList<ValueInstance> args,
		int index) : ExecutionFailed(type,
		args[index] + " at index=" + index + " does not match type=" + type + " Member=" +
		type.Members[index]);

	public sealed class CannotCallMethodWithWrongInstance(Method method, ValueInstance instance)
		: ExecutionFailed(method, instance.ToString()); //ncrunch: no coverage

	public sealed class
		TooManyArguments(Method method, string argument, IEnumerable<ValueInstance> args)
		: ExecutionFailed(method,
			argument + ", given arguments: " + args.ToWordList() + ", method " + method.Name +
			" requires these parameters: " + method.Parameters.ToWordList());

	public sealed class ArgumentDoesNotMapToMethodParameters(Method method, string message)
		: ExecutionFailed(method, message);

	public sealed class
		MissingArgument(Method method, string paramName, IEnumerable<ValueInstance> args)
		: ExecutionFailed(method,
			paramName + ", given arguments: " + args.ToWordList() + ", method " + method.Name +
			" requires these parameters: " + method.Parameters.ToWordList());

	public ValueInstance RunExpression(Expression expr, ExecutionContext context,
		bool runOnlyTests = false) =>
		expr switch
		{
			Body body => BodyEvaluator.Evaluate(body, context, runOnlyTests),
			Value v => v.Data as ValueInstance ?? CreateValueInstance(v.ReturnType, v.Data, context),
			ParameterCall or VariableCall => EvaluateVariable(expr.ToString(), context),
			MemberCall m => EvaluateMemberCall(m, context),
			ListCall listCall => MethodCallEvaluator.EvaluateListCall(listCall, context),
			If iff => IfEvaluator.Evaluate(iff, context),
			SelectorIf selectorIf => SelectorIfEvaluator.Evaluate(selectorIf, context),
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
			Expression valueExpression => RunExpression(valueExpression, context), //ncrunch: no coverage
			IList<Expression> list => list.Select(e => RunExpression(e, context)).ToList(),
			_ => valueData
		};

	private ValueInstance CreateValueInstance(Type returnType, object valueData,
		ExecutionContext context)
	{
		var value = EvaluateValueData(valueData, context);
		if (returnType.IsDictionary)
			value = NormalizeDictionaryValue(returnType, (IDictionary<string, object?>)value);
		return new ValueInstance(returnType, value);
	}

	private static object NormalizeDictionaryValue(Type dictionaryType,
		IDictionary<string, object?> rawMembers)
	{
		var listMemberName = dictionaryType.Members.First(member =>
			member.Type is GenericTypeImplementation { Generic.Name: Base.List } ||
			member.Type.Name == Base.List).Name;
		return FillDictionaryFromListKeyAndValues(rawMembers[listMemberName]);
	}

	private static object FillDictionaryFromListKeyAndValues(object? value)
	{
		var pairs = (IList)value!;
		var dictionary = new Dictionary<ValueInstance, ValueInstance>();
		foreach (var pair in pairs)
		{
			var keyAndValue = (List<ValueInstance>)((ValueInstance)pair!).Value!;
			dictionary[keyAndValue[0]] = keyAndValue[1];
		}
		return dictionary;
	}

	public class ExpressionNotSupported(Expression expr, ExecutionContext context)
		: ExecutionFailed(context.Type, expr.GetType().Name); //ncrunch: no coverage

	private static ValueInstance EvaluateVariable(string name, ExecutionContext context) =>
		context.Find(name) ?? (name == Type.ValueLowercase
			? context.This
			: null) ?? throw new ExecutionContext.VariableNotFound(name, context.Type, context.This);

	public ValueInstance EvaluateMemberCall(MemberCall member, ExecutionContext ctx)
	{
		if (ctx.This == null && ctx.Type.Members.Contains(member.Member))
			throw new UnableToCallMemberWithoutInstance(member, ctx); //ncrunch: no coverage
		if (ctx.This?.Value is Dictionary<string, object?> dict &&
			dict.TryGetValue(member.Member.Name, out var value))
			return new ValueInstance(member.ReturnType, value);
		if (member.Member.InitialValue != null && member.IsConstant)
			return RunExpression(member.Member.InitialValue, ctx);
		return member.Instance is VariableCall { Variable.Name: Type.OuterLowercase }
			? ctx.Parent!.Get(member.Member.Name)
			: ctx.Get(member.Member.Name);
	}

	internal void IncrementInlineTestDepth() => inlineTestDepth++;
	internal void DecrementInlineTestDepth() => inlineTestDepth--;

	internal ValueInstance EvaluateMethodCall(MethodCall call, ExecutionContext ctx) =>
		MethodCallEvaluator.Evaluate(call, ctx);

	public sealed class ReturnTypeMustMatchMethod(Body body, ValueInstance last) : ExecutionFailed(
		body.Method,
		"Return value " + last + " does not match method " + body.Method.Name + " ReturnType=" +
		body.Method.ReturnType);

	/// <summary>
	/// Skip parsing for trivially simple methods during validation to avoid missing-instance errors.
	/// </summary>
	private static bool IsSimpleSingleLineMethod(Method method)
	{
		if (method.lines.Count != 2)
			return false;
		var bodyLine = method.lines[1].Trim();
		var hasMethodCalls = bodyLine.Contains('(') && !bodyLine.StartsWith('(');
		if (hasMethodCalls)
			return false;
		var thenCount = CountThenSeparators(bodyLine);
		var operatorCount = bodyLine.Split(' ').Count(w => w is "and" or "or" or "not" or "is");
		return thenCount == 0 && operatorCount <= 1 || thenCount == 1 && operatorCount <= 2;
	}

	private static int CountThenSeparators(string input)
	{
		var count = 0;
		for (var index = 0; index <= input.Length - If.ThenSeparator.Length; index++)
			if (input.AsSpan(index).StartsWith(If.ThenSeparator, StringComparison.Ordinal))
			{
				count++;
				index += If.ThenSeparator.Length - 1;
			}
		return count;
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
			Not n => 1 + CountExpressionComplexity(n.Instance!), //ncrunch: no coverage
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
		Bool(not.ReturnType, !ToBool(RunExpression(not.Instance!, ctx)));

	public class UnableToCallMemberWithoutInstance(MemberCall member, ExecutionContext ctx)
		: Exception(member + ", context " + ctx); //ncrunch: no coverage

	internal static bool ToBool(object? v) =>
		v switch
		{
			bool b => b,
			ValueInstance vi => ToBool(vi.Value),
			Value { ReturnType.Name: Base.Boolean, Data: bool bv } => bv, //ncrunch: no coverage
			_ => throw new InvalidOperationException("Expected Boolean, got: " + v) //ncrunch: no coverage
		};

	private sealed class ReturnSignal(ValueInstance value) : Exception
	{
		public ValueInstance Value { get; } = value;
	}
}