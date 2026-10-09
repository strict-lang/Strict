using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

[assembly: InternalsVisibleTo("Strict.HighLevelRuntime.Tests")]
[assembly: InternalsVisibleTo("Strict.TestRunner")]

namespace Strict.HighLevelRuntime;

public partial class Interpreter
{
	public Interpreter(Package initialPackage, TestBehavior behavior = TestBehavior.OnFirstRun)
	{
		this.behavior = behavior;
		noneType = initialPackage.GetType(Type.None);
		noneInstance = new ValueInstance(noneType);
		booleanType = initialPackage.GetType(Type.Boolean);
		trueInstance = new ValueInstance(booleanType, true);
		falseInstance = new ValueInstance(booleanType, false);
		numberType = initialPackage.GetType(Type.Number);
		fileType = initialPackage.GetType(Type.File);
		characterType = initialPackage.GetType(Type.Character);
		textType = initialPackage.GetType(Type.Text);
		rangeType = initialPackage.GetType(Type.Range);
		listType = initialPackage.GetType(Type.List);
		bodyEvaluator = new BodyEvaluator(this);
		ifEvaluator = new IfEvaluator(this);
		selectorIfEvaluator = new SelectorIfEvaluator(this);
		forEvaluator = new ForEvaluator(this);
		methodCallEvaluator = new MethodCallEvaluator(this);
		toEvaluator = new ToEvaluator(this);
		directoryEvaluator = new DirectoryEvaluator(this);
	}

	internal readonly TestBehavior behavior;

	internal readonly Type noneType;

	internal readonly ValueInstance noneInstance;

	internal readonly Type booleanType;

	internal readonly ValueInstance trueInstance;

	internal readonly ValueInstance falseInstance;

	internal readonly Type numberType;

	internal readonly Type fileType;

	internal readonly Type characterType;

	internal readonly Type textType;

	internal readonly Type rangeType;

	internal readonly Type listType;

	private readonly BodyEvaluator bodyEvaluator;

	private readonly IfEvaluator ifEvaluator;

	private readonly SelectorIfEvaluator selectorIfEvaluator;

	private readonly ForEvaluator forEvaluator;

	internal readonly MethodCallEvaluator methodCallEvaluator;

	private readonly ToEvaluator toEvaluator;

	private readonly DirectoryEvaluator directoryEvaluator;

	private readonly ConcurrentStack<ExecutionContext> contextPool = new();

	internal ExecutionContext RentContext(Type type, Method method, ValueInstance? instance,
		ExecutionContext? parent)
	{
		if (contextPool.TryPop(out var ctx))
		{
			ctx.Reset(type, method, instance, parent);
			return ctx;
		}
		return new ExecutionContext(type, method, instance, parent);
	}

	internal void ReturnContext(ExecutionContext ctx) => contextPool.Push(ctx);

	internal void ResetIteration(ExecutionContext ctx)
	{
		DisposeTrackedValues(ctx);
		ctx.ResetIteration();
	}

	public ValueInstance Execute(Method method)
	{
		var returnValue = noneInstance;
		if (bodyEvaluator.InlineTestDepth == 0 && behavior != TestBehavior.Disabled &&
			(behavior == TestBehavior.TestRunner || validatedMethods.TryAdd(method, 0)))
			returnValue = Execute(method, noneInstance, [], null, true);
		if (bodyEvaluator.InlineTestDepth > 0 || behavior != TestBehavior.TestRunner)
			returnValue = Execute(method, noneInstance, []);
		return returnValue;
	}

	private readonly ConcurrentDictionary<Method, byte> validatedMethods = new();

	public readonly Statistics Statistics = new();

	public void ExecuteRunMethod(Type type)
	{
		var run = type.Methods.FirstOrDefault(m => m is { Name: Method.Run, Parameters.Count: 0 });
		if (run == null)
			throw new MethodNotFound(type, Method.Run);
		var instance = CreateFullInstance(type);
		Execute(run, instance, []);
	}

	public class MethodNotFound(Type type, string methodName)
		: InterpreterExecutionFailed(type, methodName);

	public ValueInstance Execute(Method method, ValueInstance instance, ValueInstance[] args,
		ExecutionContext? parentContext = null, bool runOnlyTests = false,
		ValueInstance[]? capturedMutableParameters = null)
	{
		Statistics.MethodCount++;
		if (parentContext is { Depth: > MaxCallDepth })
			throw new CallDepthExceeded(method, parentContext.Depth);
		args = NormalizeArguments(method, args, parentContext);
		ValidateInstanceAndArguments(method, instance, args, parentContext);
		if (TryExecuteNativeFileConstructor(method, instance, args, parentContext,
			out var fileConstructor))
			return fileConstructor;
		if (method is { Name: Method.From, Type.IsGeneric: false })
			return !instance.Equals(noneInstance)
				? throw new MethodCall.CannotCallFromConstructorWithExistingInstance()
				: !runOnlyTests && InitializesMembers(method)
					? ExecuteMemberInitializingFrom(method, args, parentContext)
					: GetFromConstructorValue(method, args);
		if (instance.TryGetValueTypeInstance()?.ReturnType.Name == Type.System)
		{ //ncrunch: no coverage start
			if (method.Name == "Write" && args.Length > 0)
				Console.WriteLine(args[0].ToExpressionCodeString());
			return noneInstance;
		} //ncrunch: no coverage end
		if (ShouldSkipGenericListTestValidation(method, runOnlyTests))
			return trueInstance;
		if (ShouldSkipKnownStrictBaseMethodValidation(method, runOnlyTests))
			return trueInstance;
		if (TryExecuteNativeFileMethod(method, instance, args, out var fileResult))
			return fileResult;
		if (directoryEvaluator.TryEvaluate(method, args, out var directoryResult))
			return directoryResult;
		if (method is { Name: NativeProcessRunner.OperatingSystemMethod, Type.Name: "Process" })
			return new ValueInstance(NativeProcessRunner.OperatingSystemName);
		if (method is { Name: "Find", Type.Name: "Process" } && args.Length == 1)
			return new ValueInstance(method.Type,
				[new ValueInstance(NativeProcessRunner.FindTool(args[0].Text) ?? "")]);
		if (TryExecuteTextWriterWrite(method, args))
			return noneInstance;
		if (runOnlyTests && IsSimpleSingleLineMethod(method))
			return trueInstance;
		var context = CreateExecutionContext(method, instance, args, parentContext, runOnlyTests);
		try
		{
			Expression body;
			try
			{
				body = method.GetBodyAndParseIfNeeded(runOnlyTests && method.Type.IsGeneric);
			}
			catch (Exception inner) when (runOnlyTests)
			{
				if (ShouldIgnoreGenericListTestParseFailure(method, inner))
					return trueInstance;
				throw new MethodRequiresTest(method,
					$"Test execution failed: {method.Parent.FullName}.{method.Name}\n" +
					method.lines.ToLines() + Environment.NewLine + inner);
			}
			if (body is not Body && runOnlyTests)
				return method.Name == Method.Run
					? noneInstance
					: IsSimpleExpressionWithLessThanThreeSubExpressions(body)
						? trueInstance
						: throw new MethodRequiresTest(method, body.ToString());
			var result = RunExpression(body, context, runOnlyTests);
			if (capturedMutableParameters != null)
				CaptureMutableParameterValues(method, context, capturedMutableParameters);
			return context.ExitMethodAndReturnValue ??
				result.ApplyMethodReturnTypeMutable(method.ReturnType);
		}
		finally
		{
			DisposeTrackedValues(context);
			ReturnContext(context);
		}
	}

	private static void CaptureMutableParameterValues(Method method, ExecutionContext context,
		ValueInstance[] capturedMutableParameters)
	{
		for (var index = 0; index < method.Parameters.Count && index < capturedMutableParameters.Length;
			index++)
		{
			var parameter = method.Parameters[index];
			if (parameter.IsMutable && context.Variables.TryGetValue(parameter.Name, out var finalValue))
				capturedMutableParameters[index] = finalValue;
		}
	}

	private ExecutionContext CreateExecutionContext(Method method, ValueInstance instance,
		IReadOnlyList<ValueInstance> args, ExecutionContext? parentContext, bool runOnlyTests)
	{
		var context = RentContext(method.Type, method, instance, parentContext);
		for (var index = 0; index < method.Parameters.Count; index++)
		{
			var parameter = method.Parameters[index];
			var argument = index < args.Count
				? args[index]
				: parameter.DefaultValue != null
					? RunExpression(parameter.DefaultValue, context)
					: runOnlyTests
						? TryAutoCreateInstance(parameter.Type) ?? GetDefaultValue(parameter.Type)
						: throw new MissingArgument(method, parameter.Name, args);
			context.Variables[parameter.Name] = argument;
		}
		return context;
	}

	private ValueInstance[] NormalizeArguments(Method method, ValueInstance[] args,
		ExecutionContext? parentContext)
	{
		if (args.Length == 0)
			return args;
		ValueInstance[]? normalizedArgs = null;
		for (var index = 0; index < args.Length && index < method.Parameters.Count; index++)
		{
			var parameterType = method.Parameters[index].Type;
			if (args[index].IsSameOrCanBeUsedAs(parameterType) || parameterType.IsIterator ||
				IsSingleCharacterTextArgument(parameterType, args[index]))
				continue;
			var normalizedArgument = TryConvertListArgument(args[index], parameterType, parentContext);
			if (normalizedArgument is not { } convertedArgument)
				continue;
			normalizedArgs ??= (ValueInstance[])args.Clone();
			normalizedArgs[index] = convertedArgument;
		}
		return normalizedArgs ?? args;
	}

	private ValueInstance? TryConvertListArgument(ValueInstance argument, Type parameterType,
		ExecutionContext? parentContext)
	{
		if (!argument.IsList || !parameterType.IsList)
			return null;
		var targetItemType = parameterType.GetFirstImplementation();
		var items = argument.List.Items;
		var convertedItems = new ValueInstance[items.Count];
		for (var index = 0; index < items.Count; index++)
		{
			var convertedItem = TryConvertListItem(items[index], targetItemType, parentContext);
			if (convertedItem is not { } convertedListItem)
				return null;
			convertedItems[index] = convertedListItem;
		}
		return new ValueInstance(parameterType, convertedItems);
	}

	private ValueInstance? TryConvertListItem(ValueInstance item, Type targetType,
		ExecutionContext? parentContext)
	{
		if (item.IsSameOrCanBeUsedAs(targetType))
			return item;
		if (item.IsList && targetType.IsList)
			return TryConvertListArgument(item, targetType, parentContext);
		var sourceType = item.TryGetValueTypeInstance()?.ReturnType ?? item.GetType();
		if (!sourceType.CanBeConvertedTo(targetType))
			return null;
		if (sourceType.AvailableMethods.TryGetValue(BinaryOperator.To, out var toMethods))
		{
			var toMethod = toMethods.FirstOrDefault(method => method.ReturnType == targetType ||
				method.ReturnType.IsSameOrCanBeUsedAs(targetType, false));
			if (toMethod != null)
				return Execute(toMethod, item, [], parentContext);
		}
		if (!targetType.AvailableMethods.TryGetValue(Method.From, out var fromMethods))
			return null;
		var fromMethod = fromMethods.FirstOrDefault(method => method.Parameters.Count == 1 &&
			sourceType.IsSameOrCanBeUsedAs(method.Parameters[0].Type, false));
		return fromMethod != null
			? Execute(fromMethod, noneInstance, [item], parentContext)
			: null;
	}

	private void ValidateInstanceAndArguments(Method method, ValueInstance instance,
		IReadOnlyList<ValueInstance> args, ExecutionContext? parentContext)
	{
		if (!instance.IsPrimitiveType(noneType) && !instance.IsSameOrCanBeUsedAs(method.Type))
			throw new CannotCallMethodWithWrongInstance(method, instance, method.Type);
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
		ThrowIfSameMethodCallExistsInParentChain(method, instance, args, parentContext);
	}

	private void ThrowIfSameMethodCallExistsInParentChain(Method method, ValueInstance instance,
		IReadOnlyList<ValueInstance> args, ExecutionContext? parentContext)
	{
		if (HasMutableParameter(method))
			return;
		for (var current = parentContext; current != null; current = current.Parent)
			if (current.Method == method && !current.IsTestAtCurrentLine &&
				AreSameInstanceForRecursionCheck(current.This, instance) &&
				DoArgumentsMatch(method, args, current.Variables))
				throw new StackOverflowCallingItselfWithSameInstanceAndArguments(method, instance, args,
					current);
	}

	/// <summary>
	/// Mutable arguments are the same objects in every call but change in between (e.g. shrinking a
	/// list), equal arguments do not mean endless recursion then.
	/// </summary>
	private static bool HasMutableParameter(Method method)
	{
		for (var index = 0; index < method.Parameters.Count; index++)
			if (method.Parameters[index].IsMutable)
				return true;
		return false;
	}

	private bool AreSameInstanceForRecursionCheck(ValueInstance? parentThis,
		ValueInstance currentInstance)
	{
		if (!parentThis.HasValue)
			return currentInstance.Equals(noneInstance);
		var parent = parentThis.Value;
		return parent.IsPrimitiveType(noneType)
			? currentInstance.IsPrimitiveType(noneType)
			: parent.Equals(currentInstance);
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
		ValueInstance? instance,
		IReadOnlyList<ValueInstance> args,
		ExecutionContext parentContext) : InterpreterExecutionFailed(method,
		"Stack overflow detected while calling " + FormatCall(method, instance, args) +
		". Matching parent call chain: " + FormatParentChain(parentContext))
	{
		private static string FormatCall(Method method, ValueInstance? instance,
			IReadOnlyList<ValueInstance> args) =>
			method + ", instance=" + (instance?.ToString() ?? Type.None) + ", arguments=" +
			args.ToBrackets();

		private static string FormatParentChain(ExecutionContext context)
		{
			var callChain = "";
			for (var current = context; current != null; current = current.Parent)
				callChain += (callChain.Length == 0
						? ""
						: " -> ") + current.Method + ", instance=" + (current.This?.ToString() ?? Type.None) +
					", arguments=" + FormatArguments(current.Method, current.Variables);
			return callChain;
		}

		private static string FormatArguments(Method method,
			IReadOnlyDictionary<string, ValueInstance> variables)
		{
			var arguments = "";
			for (var index = 0; index < method.Parameters.Count; index++)
				if (variables.TryGetValue(method.Parameters[index].Name, out var value))
					arguments += (arguments.Length == 0
						? "("
						: ", ") + value;
			return arguments.Length == 0
				? "()"
				: arguments + ")";
		}
	}

	private static readonly IReadOnlyDictionary<string, string> TraitImplementationRegistry =
		new Dictionary<string, string> { { Type.TextWriter, Type.System } };

	public sealed class
		InvalidTypeForArgument(Type type, IReadOnlyList<ValueInstance> args, int index)
		: InterpreterExecutionFailed(type,
			args[index] + " at index=" + index + " does not match type=" + type + " Member=" +
			type.Members[index]);

	public sealed class CannotCallMethodWithWrongInstance(Method method,
		ValueInstance instance,
		Type expectedInstanceType) : InterpreterExecutionFailed(method,
		instance + " is wrong, expected: " + expectedInstanceType);

	public sealed class
		TooManyArguments(Method method, string argument, IReadOnlyList<ValueInstance> args)
		: InterpreterExecutionFailed(method,
			argument + ", given arguments: " + string.Join(", ", args) + ", method " + method.Name +
			" requires these parameters: " + string.Join(", ", method.Parameters));

	public sealed class ArgumentDoesNotMapToMethodParameters(Method method, string message)
		: InterpreterExecutionFailed(method, message);

	public sealed class
		MissingArgument(Method method, string paramName, IReadOnlyList<ValueInstance> args)
		: InterpreterExecutionFailed(method,
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
			Dictionary dict => dict.Data,
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
			MethodCall call => methodCallEvaluator.Evaluate(call, context),
			Declaration c => EvaluateAndAssign(c.Name, c.Value, context, true),
			MutableReassignment a => a.Target is ListCall listCallTarget
				? EvaluateMutableListElementAssignment(listCallTarget, a.Value, context)
				: EvaluateAndAssign(a.Name, a.Value, context, false),
			Instance => EvaluateVariable(Type.ValueLowercase, context),
			_ => throw new ExpressionNotSupported(expr, context) //ncrunch: no coverage
		};
	}

	/// <summary>
	/// Endless recursion must fail with a Strict error, not crash the process with a stack overflow.
	/// </summary>
	private const int MaxCallDepth = 128;

	public sealed class CallDepthExceeded(Method method, int depth) : InterpreterExecutionFailed(method,
		"Call depth " + depth + " exceeded " + MaxCallDepth + ", endless recursion?");

	public class ExpressionNotSupported(Expression expr, ExecutionContext context)
		: InterpreterExecutionFailed(context.Type, expr.GetType().Name); //ncrunch: no coverage

	public sealed class ReturnTypeMustMatchMethod(Body body, ValueInstance last)
		: InterpreterExecutionFailed(body.Method,
			"Return value " + last + " does not match method " + body.Method.Name + " ReturnType=" +
			body.Method.ReturnType);

	private readonly ConcurrentDictionary<Method, bool> simpleMethodCache = new();

	private const int MaxSimpleExpressionComplexity = 3;

	public class MethodRequiresTest(Method method, string body) : InterpreterExecutionFailed(method,
		body.StartsWith("Test execution failed", StringComparison.Ordinal)
			? body
			: $"Method {method.Parent.FullName}.{method.Name}\n{body}")
	{
		public MethodRequiresTest(Method method, Body body) : this(method,
			body + " ({CountExpressionComplexity(body)} expressions)") { }
	}

	public sealed class TestFailed(Method method,
		Expression expression,
		ValueInstance result,
		string details) : InterpreterExecutionFailed(method,
		$"\"{method.Name}\" method failed: {expression}, result: {result}" + (details.Length > 0
			? $", evaluated: {details}"
			: "") + " in" + Environment.NewLine +
		$"{method.Type.FilePath}:line {expression.LineNumber + 1}")
	{
		public Expression FailedExpression { get; } = expression;
		public ValueInstance Result { get; } = result;
		public string Details { get; } = details;
	}

	public ValueInstance ToBoolean(bool isTrue) =>
		isTrue
			? trueInstance
			: falseInstance;
}
