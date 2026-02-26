using Strict.Expressions;
using Strict.Language;
using System.Collections;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

public sealed class Executor
{
	public Executor(Package initialPackage, TestBehavior behavior = TestBehavior.OnFirstRun)
	{
		this.behavior = behavior;
		noneType = initialPackage.GetType(Base.None);
		noneInstance = new ValueInstance(noneType);
		booleanType = initialPackage.GetType(Base.Boolean);
		trueInstance = new ValueInstance(booleanType, true);
		falseInstance = new ValueInstance(booleanType, false);
		numberType = initialPackage.GetType(Base.Number);
		bodyEvaluator = new BodyEvaluator(this);
		ifEvaluator = new IfEvaluator(this);
		selectorIfEvaluator = new SelectorIfEvaluator(this);
		forEvaluator = new ForEvaluator(this);
		methodCallEvaluator = new MethodCallEvaluator(this);
		toEvaluator = new ToEvaluator(this);
	}

	private readonly TestBehavior behavior;
	private readonly Type noneType;
	private readonly ValueInstance noneInstance;
	private readonly Type booleanType;
	private readonly ValueInstance trueInstance;
	private readonly ValueInstance falseInstance;
	private readonly Type numberType;
	private readonly BodyEvaluator bodyEvaluator;
	private readonly IfEvaluator ifEvaluator;
	private readonly SelectorIfEvaluator selectorIfEvaluator;
	private readonly ForEvaluator forEvaluator;
	private readonly MethodCallEvaluator methodCallEvaluator;
	private readonly ToEvaluator toEvaluator;

	public ValueInstance Execute(Method method)
	{
		var returnValue = noneInstance;
		if (inlineTestDepth == 0 && behavior != TestBehavior.Disabled &&
			(behavior == TestBehavior.TestRunner || validatedMethods.Add(method)))
			returnValue = Execute(method, noneInstance, Array.Empty<ValueInstance>(), null, true);
		if (inlineTestDepth > 0 || behavior != TestBehavior.TestRunner)
			returnValue = Execute(method, noneInstance, Array.Empty<ValueInstance>(), null, false);
		return returnValue;
	}

	/// <summary>
	/// Evaluate inline tests at top-level only (outermost call), avoid recursion
	/// </summary>
	private int inlineTestDepth;
	private readonly HashSet<Method> validatedMethods = [];
	public Statistics Statistics { get; } = new();

	private ValueInstance Execute(Method method, ValueInstance instance,
		IReadOnlyList<ValueInstance> args, ExecutionContext? parentContext, bool runOnlyTests = false)
	{
		if (runOnlyTests)
			Statistics.MethodTested++;
		else
			Statistics.MethodCount++;
		ValidateInstanceAndArguments(method, instance, args, parentContext);
		if (method is { Name: Method.From, Type.IsGeneric: false })
			return instance != noneInstance
				? throw new MethodCall.CannotCallFromConstructorWithExistingInstance()
				: CreateValueInstance(method.Type, GetFromConstructorValue(method, args));
		var context = CreateExecutionContext(method, instance, args, parentContext, runOnlyTests);
		if (runOnlyTests && IsSimpleSingleLineMethod(method))
			return trueInstance;
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
				? trueInstance
				: throw new MethodRequiresTest(method, body.ToString());
		var result = RunExpression(body, context, runOnlyTests);
		if (context.ExitMethodAndReturnValue.HasValue)
			return context.ExitMethodAndReturnValue.Value;
		if (runOnlyTests)
			return result;
		return result.ApplyMethodReturnTypeMutable(method.ReturnType);
	}

	private ExecutionContext CreateExecutionContext(Method method, ValueInstance instance,
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
		context.AddDictionaryElements(instance);
		return context;
	}

	private static void ValidateInstanceAndArguments(Method method, ValueInstance? instance,
		IReadOnlyList<ValueInstance> args, ExecutionContext? parentContext)
	{
		if (instance.HasValue && !instance.Value.ReturnType.IsSameOrCanBeUsedAs(method.Type))
			throw new CannotCallMethodWithWrongInstance(method, instance.Value);
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
			(parentContext.This?.Equals(instance) ?? !instance.HasValue) &&
			DoArgumentsMatch(method, args, parentContext.Variables))
			throw new StackOverflowCallingItselfWithSameInstanceAndArguments(method, instance, args,
				parentContext);
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

	private object? GetFromConstructorValue(Method method, IReadOnlyList<ValueInstance> args)
	{
		Statistics.FromCreationsCount++;
		if (args.Count == 0 && method.Type.Name == Base.Text)
			return "";
		if (method.Type.IsDictionary && args is [{ ReturnType.IsIterator: true }])
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
		bool runOnlyTests = false)
	{
		Statistics.ExpressionCount++;
		return expr switch
		{
			Body body => bodyEvaluator.Evaluate(body, context, runOnlyTests),
			Value v => v.Data is ValueInstance valueInstance
				? valueInstance
				: CreateValueInstance(v.ReturnType, v.Data, context),
			ParameterCall or VariableCall => EvaluateVariable(expr.ToString(), context),
			MemberCall m => EvaluateMemberCall(m, context),
			ListCall listCall => methodCallEvaluator.EvaluateListCall(listCall, context),
			If iff => ifEvaluator.Evaluate(iff, context),
			SelectorIf selectorIf => selectorIfEvaluator.Evaluate(selectorIf, context),
			For f => forEvaluator.Evaluate(f, context),
			Return r => EvaluateReturn(r, context),
			To t => toEvaluator.Evaluate(t, context),
			Not n => EvaluateNot(n, context),
			MethodCall call => EvaluateMethodCall(call, context),
			Declaration c => EvaluateAndAssign(c.Name, c.Value, context, true),
			MutableReassignment a => EvaluateAndAssign(a.Name, a.Value, context, false),
			Instance => EvaluateVariable(Type.ValueLowercase, context),
			_ => throw new ExpressionNotSupported(expr, context) //ncrunch: no coverage
		};
	}

	private object EvaluateValueData(object valueData, ExecutionContext context) =>
		valueData switch
		{
			Expression valueExpression =>
				RunExpression(valueExpression, context), //ncrunch: no coverage
			IList<Expression> list => list.Select(e => RunExpression(e, context)).ToList(),
			_ => valueData
		};

	private ValueInstance CreateValueInstance(Type returnType, object valueData,
		ExecutionContext context)
	{
		var value = EvaluateValueData(valueData, context);
		if (returnType.IsDictionary)
			value = NormalizeDictionaryValue(returnType, (IDictionary<string, object?>)value);
		return CreateValueInstance(returnType, value);
	}

	internal ValueInstance CreateValueInstance(Type returnType, object? value)
	{
		if (value is ValueInstance valueInstance)
			return CreateValueInstance(returnType, valueInstance);
		return Make(ValueInstance.Create(returnType, value));
	}

	internal ValueInstance CreateValueInstance(Type returnType, ValueInstance value) =>
		Make(ValueInstance.Create(returnType, value));

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

	private ValueInstance EvaluateVariable(string name, ExecutionContext context)
	{
		Statistics.VariableCallCount++;
		return context.Find(name, Statistics) ?? (name == Type.ValueLowercase
			? context.This
			: null) ?? throw new ExecutionContext.VariableNotFound(name, context.Type, context.This);
	}

	public ValueInstance EvaluateMemberCall(MemberCall member, ExecutionContext ctx)
	{
		Statistics.MemberCallCount++;
		if (ctx.This == null && ctx.Type.Members.Contains(member.Member))
			throw new UnableToCallMemberWithoutInstance(member, ctx); //ncrunch: no coverage
		if (ctx.This?.Value is Dictionary<string, object?> dict &&
			dict.TryGetValue(member.Member.Name, out var value))
			return CreateValueInstance(member.ReturnType, value);
		if (member.Member.InitialValue != null && member.IsConstant)
			return RunExpression(member.Member.InitialValue, ctx);
		return member.Instance is VariableCall { Variable.Name: Type.OuterLowercase }
			? ctx.Parent!.Get(member.Member.Name, Statistics)
			: ctx.Get(member.Member.Name, Statistics);
	}

	internal void IncrementInlineTestDepth() => inlineTestDepth++;
	internal void DecrementInlineTestDepth() => inlineTestDepth--;

	internal ValueInstance EvaluateMethodCall(MethodCall call, ExecutionContext ctx) =>
		methodCallEvaluator.Evaluate(call, ctx);

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

	public sealed class TestFailed(
		Method method,
		Expression expression,
		ValueInstance result,
		string details) : ExecutionFailed(method,
		$"\"{method.Name}\" method failed: {expression}, result: {result}" + (details.Length > 0
			? $", evaluated: {details}"
			: "") + " in" + Environment.NewLine +
		$"{method.Type.FilePath}:line {expression.LineNumber + 1}");

	private ValueInstance EvaluateAndAssign(string name, Expression value, ExecutionContext ctx,
		bool isDeclaration)
	{
		if (isDeclaration)
			Statistics.VariableDeclarationCount++;
		if (value.IsMutable)
		{
			if (isDeclaration)
				Statistics.MutableDeclarationCount++;
			Statistics.MutableUsageCount++;
		}
		return ctx.Set(name, RunExpression(value, ctx));
	}

	private ValueInstance EvaluateReturn(Return r, ExecutionContext ctx)
	{
		Statistics.ReturnCount++;
		var result = RunExpression(r.Value, ctx);
		ctx.ExitMethodAndReturnValue = result;
		return result;
	}

	private ValueInstance EvaluateNot(Not not, ExecutionContext ctx)
	{
		Statistics.UnaryCount++;
		return Bool(not.ReturnType, !ToBool(RunExpression(not.Instance!, ctx)));
	}

	public class UnableToCallMemberWithoutInstance(MemberCall member, ExecutionContext ctx)
		: Exception(member + ", context " + ctx); //ncrunch: no coverage

	internal static bool ToBool(object? v) =>
		v switch
		{
			bool b => b,
			ValueInstance vi => vi.AsBool(),
			Value { ReturnType.Name: Base.Boolean, Data: bool bv } => bv, //ncrunch: no coverage
			_ => throw new InvalidOperationException("Expected Boolean, got: " +
				v) //ncrunch: no coverage
		};

	/*nah
	private void EnsureCachedBaseValues(Context context)
	{
		if (cachedValuesInitialized)
			return;
		cachedValuesInitialized = true;
		//TODO: cache types here instead
		cachedTrue = Make(ValueInstance.CreateBoolean(context, true));
		cachedFalse = Make(ValueInstance.CreateBoolean(context, false));
		cachedNone = Make(ValueInstance.CreateNone(context));
		cachedNumbers = new ValueInstance[12];
		for (var value = -1; value <= 10; value++)
			cachedNumbers[value + 1] = Make(ValueInstance.CreateNumber(context, value));
	}

	/// <summary>
	/// Single entry point for creating ValueInstances in HighLevelRuntime.
	/// Updates Statistics and returns the instance. All code should use this instead of
	/// calling ValueInstance constructors or Create* methods directly.
	/// </summary>
	private ValueInstance Make(ValueInstance instance)
	{
		Statistics.ValueInstanceCount++;
		var effectiveType = ValueInstance.GetEffectiveType(instance.ReturnType);
		if (effectiveType.IsBoolean)
			Statistics.BooleanCount++;
		else if (effectiveType.Name == Base.Number)
			Statistics.NumberCount++;
		else if (ValueInstance.IsTextType(effectiveType))
			Statistics.TextCount++;
		else if (effectiveType.IsList)
			Statistics.ListCount++;
		else if (effectiveType.IsDictionary)
			Statistics.DictionaryCount++;
		return instance;
	}

	private ValueInstance? cachedTrue;
	private ValueInstance? cachedFalse;
	private ValueInstance? cachedNone;
	private ValueInstance[] cachedNumbers = Array.Empty<ValueInstance>();
	private bool cachedValuesInitialized;

	internal ValueInstance Bool(Context any, bool b)
	{
		EnsureCachedBaseValues(any);
		return b
			? cachedTrue!.Value
			: cachedFalse!.Value;
	}

	internal ValueInstance None(Context any)
	{
		EnsureCachedBaseValues(any);
		return cachedNone!.Value;
	}

	internal ValueInstance Number(Context any, double number)
	{
		EnsureCachedBaseValues(any);
		if (double.IsInteger(number))
		{
			var index = (int)number + 1;
			if ((uint)index < (uint)cachedNumbers.Length)
				return cachedNumbers[index];
		}
		return Make(ValueInstance.CreateNumber(any, number));
	}

	private static bool IsNumberType(Type type) => ValueInstance.IsNumberType(type);
	private static bool IsTextType(Type type) => type.Name is Base.Text or Base.Name;

	private static Type GetEffectiveType(Type returnType) =>
		returnType.IsMutable
			? ((GenericTypeImplementation)returnType).ImplementationTypes[0]
			: returnType;
	*/
}