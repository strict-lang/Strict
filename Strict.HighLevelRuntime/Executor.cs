using Strict.Expressions;
using Strict.Language;
using System.Runtime.CompilerServices;
using Type = Strict.Language.Type;

[assembly: InternalsVisibleTo("Strict.HighLevelRuntime.Tests")]
[assembly: InternalsVisibleTo("Strict.TestRunner")]

namespace Strict.HighLevelRuntime;

public class Executor
{
	public Executor(Package initialPackage, TestBehavior behavior = TestBehavior.OnFirstRun)
	{
		this.behavior = behavior;
		noneType = initialPackage.GetType(Type.None);
		noneInstance = new ValueInstance(noneType);
		booleanType = initialPackage.GetType(Type.Boolean);
		trueInstance = new ValueInstance(booleanType, true);
		falseInstance = new ValueInstance(booleanType, false);
		numberType = initialPackage.GetType(Type.Number);
		characterType = initialPackage.GetType(Type.Character);
		rangeType = initialPackage.GetType(Type.Range);
		listType = initialPackage.GetType(Type.List);
		bodyEvaluator = new BodyEvaluator(this);
		ifEvaluator = new IfEvaluator(this);
		selectorIfEvaluator = new SelectorIfEvaluator(this);
		forEvaluator = new ForEvaluator(this);
		methodCallEvaluator = new MethodCallEvaluator(this);
		toEvaluator = new ToEvaluator(this);
	}

	private readonly TestBehavior behavior;
	internal readonly Type noneType;
	internal readonly ValueInstance noneInstance;
	internal readonly Type booleanType;
	internal readonly ValueInstance trueInstance;
	internal readonly ValueInstance falseInstance;
	internal readonly Type numberType;
	internal readonly Type characterType;
	internal readonly Type rangeType;
	internal readonly Type listType;
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
			returnValue = Execute(method, noneInstance, [], null, true);
		if (inlineTestDepth > 0 || behavior != TestBehavior.TestRunner)
			returnValue = Execute(method, noneInstance, []);
		return returnValue;
	}

	/// <summary>
	/// Evaluate inline tests at top-level only (outermost call), avoid recursion
	/// </summary>
	private int inlineTestDepth;
	private readonly HashSet<Method> validatedMethods = [];
	public Statistics Statistics { get; } = new();

	public ValueInstance Execute(Method method, ValueInstance instance,
		IReadOnlyList<ValueInstance> args, ExecutionContext? parentContext = null, bool runOnlyTests = false)
	{
		Statistics.MethodCount++;
		ValidateInstanceAndArguments(method, instance, args, parentContext);
		if (method is { Name: Method.From, Type.IsGeneric: false })
			return instance.Equals(noneInstance)
				? GetFromConstructorValue(method, args)
				: throw new MethodCall.CannotCallFromConstructorWithExistingInstance();
		if (runOnlyTests && IsSimpleSingleLineMethod(method))
			return trueInstance;
		var context = CreateExecutionContext(method, instance, args, parentContext, runOnlyTests);
		Expression body;
		try
		{
			body = method.GetBodyAndParseIfNeeded(runOnlyTests && method.Type.IsGeneric);
		}
		catch (Exception inner) when (runOnlyTests)
		{
			throw new MethodRequiresTest(method,
				$"Test execution failed: {method.Parent.FullName}.{method.Name}\n" +
				method.lines.ToLines() + Environment.NewLine + inner);
		}
		if (body is not Body && runOnlyTests)
			return IsSimpleExpressionWithLessThanThreeSubExpressions(body)
				? trueInstance
				: throw new MethodRequiresTest(method, body.ToString());
		var result = RunExpression(body, context, runOnlyTests);
		return context.ExitMethodAndReturnValue ??
			result.ApplyMethodReturnTypeMutable(method.ReturnType);
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
		return context;
	}

	private void ValidateInstanceAndArguments(Method method, ValueInstance instance,
		IReadOnlyList<ValueInstance> args, ExecutionContext? parentContext)
	{
		if (!instance.IsPrimitiveType(noneType) && !instance.IsSameOrCanBeUsedAs(method.Type))
			throw new CannotCallMethodWithWrongInstance(method, instance);
		if (args.Count > method.Parameters.Count)
			throw new TooManyArguments(method, args[method.Parameters.Count].ToString(), args);
		for (var index = 0; index < args.Count; index++)
			if (!args[index].IsSameOrCanBeUsedAs(method.Parameters[index].Type) &&
				!method.Parameters[index].Type.IsIterator && method.Name != Method.From &&
				!IsSingleCharacterTextArgument(method.Parameters[index].Type, args[index]))
				throw new ArgumentDoesNotMapToMethodParameters(method,
					"Method \"" + method + "\" parameter " + index + ": " +
					method.Parameters[index].ToStringWithInnerMembers() +
					" cannot be assigned from argument " + args[index]);
		if (parentContext != null && parentContext.Method == method &&
			(parentContext.This?.Equals(instance) ?? instance.Equals(noneInstance)) &&
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
		ValueInstance? instance, IReadOnlyList<ValueInstance> args, ExecutionContext parentContext)
		: ExecutionFailed(method, "Parent context=" + parentContext + ", Instance=" + instance +
			", arguments=" + string.Join(", ", args));

	private ValueInstance GetFromConstructorValue(Method method, IReadOnlyList<ValueInstance> args)
	{
		Statistics.FromCreationsCount++;
		if (args.Count == 0 && method.Type.IsText)
			return new ValueInstance("");
		if ((method.Type.IsCharacter || method.Type.IsNumber || method.Type.IsEnum) && args.Count == 1)
		{
			if (IsSingleCharacterTextArgument(method.Type, args[0]))
				return new ValueInstance(method.Type, args[0].Text[0]);
			if (!args[0].IsText || args[0].IsSameOrCanBeUsedAs(method.Type))
				return new ValueInstance(method.Type, args[0].Number);
		} //ncrunch: no coverage
		if (method.Type.IsList)
			return new ValueInstance(method.Type, args);
		if (method.Type.IsDictionary)
			return args[0].IsDictionary
				? args[0]
				: new ValueInstance(method.Type, FillDictionaryFromListKeyAndValues(args[0]));
		var members = new Dictionary<string, ValueInstance>(StringComparer.Ordinal);
		for (var index = 0; index < args.Count; index++)
		{
			var parameter = method.Parameters[index];
			if (!args[index].IsSameOrCanBeUsedAs(parameter.Type) &&
				!parameter.Type.IsIterator && !IsSingleCharacterTextArgument(parameter.Type, args[index]))
				throw new InvalidTypeForArgument(method.Type, args, index);
			var memberName = GetFirstMemberMatchingParameter(method, parameter)?.Name ??
				parameter.Name.MakeFirstLetterUppercase();
			members.Add(memberName, IsSingleCharacterTextArgument(parameter.Type, args[index])
				? new ValueInstance(characterType, args[index].Text[0])
				: args[index]);
		}
		return new ValueInstance(method.Type, members);
	}

	private static Member? GetFirstMemberMatchingParameter(Method method, Parameter parameter)
	{
		foreach (var member in method.Type.Members)
			if (member.Name.Equals(parameter.Name, StringComparison.OrdinalIgnoreCase))
				return member;
		return null; //ncrunch: no coverage
	}

	private static Dictionary<ValueInstance, ValueInstance> FillDictionaryFromListKeyAndValues(
		ValueInstance value)
	{
		var dictionary = new Dictionary<ValueInstance, ValueInstance>();
		foreach (var pair in value.List.Items)
		{
			var keyAndValue = pair.List.Items;
			dictionary[keyAndValue[0]] = keyAndValue[1];
		}
		return dictionary;
	}

	private static bool IsSingleCharacterTextArgument(Type targetType, ValueInstance value) =>
		value is { IsText: true, Text.Length: 1 } && (targetType.IsNumber || targetType.IsCharacter);

	public sealed class InvalidTypeForArgument(Type type, IReadOnlyList<ValueInstance> args,
		int index) : ExecutionFailed(type, args[index] + " at index=" + index + " does not match " +
		"type=" + type + " Member=" + type.Members[index]);

	public sealed class CannotCallMethodWithWrongInstance(Method method, ValueInstance instance)
		: ExecutionFailed(method, instance.ToString()); //ncrunch: no coverage

	public sealed class TooManyArguments(Method method, string argument,
		IReadOnlyList<ValueInstance> args) : ExecutionFailed(method,
		argument + ", given arguments: " + string.Join(", ", args) + ", method " + method.Name +
		" requires these parameters: " + string.Join(", ", method.Parameters));

	public sealed class ArgumentDoesNotMapToMethodParameters(Method method, string message)
		: ExecutionFailed(method, message);

	public sealed class MissingArgument(Method method, string paramName,
		IReadOnlyList<ValueInstance> args) : ExecutionFailed(method,
		paramName + ", given arguments: " + string.Join(", ", args) + ", method " + method.Name +
		" requires these parameters: " + string.Join(", ", method.Parameters));

	public ValueInstance RunExpression(Expression expr, ExecutionContext context,
		bool runOnlyTests = false)
	{
		Statistics.ExpressionCount++;
		return expr switch
		{
			Body body => bodyEvaluator.Evaluate(body, context, runOnlyTests),
			List list => EvaluateListExpression(list, context),
			//TODO: add test first: Dictionary dict =>
			Value v => v.Data,
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

	private ValueInstance EvaluateListExpression(List list, ExecutionContext context)
	{
		var values = new List<ValueInstance>(list.Values.Count);
		foreach (var value in list.Values)
			values.Add(RunExpression(value, context));
		return new ValueInstance(list.ReturnType, values);
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
		if (ctx.This is { IsDictionary: true } &&
			member.Member.Name.Equals(Type.ElementsLowercase, StringComparison.OrdinalIgnoreCase))
		{
			var pairs = new List<ValueInstance>();
			var pairType = member.Member.Type is { IsList: true, IsGeneric: true }
				? listType.GetFirstImplementation()
				: member.Member.Type;
			foreach (var pair in ctx.This.Value.GetDictionaryItems())
				pairs.Add(new ValueInstance(pairType, [pair.Key, pair.Value]));
			return new ValueInstance(member.Member.Type, pairs);
		}
		var typeInstance = ctx.This?.TryGetValueTypeInstance();
		if (typeInstance != null && typeInstance.Members.TryGetValue(member.Member.Name, out var value))
			return value;
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
		body.Method, "Return value " + last + " does not match method " + body.Method.Name +
		" ReturnType=" + body.Method.ReturnType);

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

	public sealed class TestFailed(Method method,	Expression expression, ValueInstance result,
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
		return !RunExpression(not.Instance!, ctx).Boolean
			? trueInstance
			: falseInstance;
	}

	public class UnableToCallMemberWithoutInstance(MemberCall member, ExecutionContext ctx)
		: Exception(member + ", context " + ctx); //ncrunch: no coverage

	public ValueInstance ToBoolean(bool isTrue) =>
		isTrue
			? trueInstance
			: falseInstance;
}