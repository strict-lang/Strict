using Strict.Expressions;
using Strict.Language;
using System.Collections;
using System.Globalization;
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
			var listMemberName = method.Type.Members.FirstOrDefault(member =>
				member.Type is GenericTypeImplementation { Generic.Name: Base.List } ||
				member.Type.Name == Base.List)?.Name ?? Type.ElementsLowercase;
			return new Dictionary<string, object?>(StringComparer.Ordinal)
			{
				[listMemberName] = args[0].Value
			};
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
			var listMember = type.Members.FirstOrDefault(member =>
				member.Type is GenericTypeImplementation { Generic.Name: Base.List } ||
				member.Type.Name == Base.List);
			return new Dictionary<string, object?>(StringComparer.Ordinal)
			{
				[listMember?.Name ?? Type.ElementsLowercase] = new List<ValueInstance>()
			};
		}
		return null;
	}

	private static void AddDictionaryElementsAlias(ExecutionContext context,
		ValueInstance? instance)
	{
		if (instance?.ReturnType is not GenericTypeImplementation { Generic.Name: Base.Dictionary })
			return;
		if (instance.Value is not IDictionary<string, object?> rawMembers ||
			context.Variables.ContainsKey(Type.ElementsLowercase))
			return;
		var listValue = rawMembers.TryGetValue("elements", out var elementsValue)
			? elementsValue
			: rawMembers.TryGetValue(Type.ElementsLowercase, out var keyValues)
				? keyValues
				: rawMembers.Values.FirstOrDefault();
		if (listValue != null)
			context.Set(Type.ElementsLowercase,
				new ValueInstance(instance.ReturnType.GetType(Base.List), listValue));
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
			Body body => EvaluateBody(body, context, runOnlyTests),
			Value v => v.Data as ValueInstance ??
				new ValueInstance(v.ReturnType, EvaluateValueData(v.Data, context)),
			ParameterCall or VariableCall => EvaluateVariable(expr.ToString(), context),
			MemberCall m => EvaluateMember(m, context),
			ListCall listCall => EvaluateListCall(listCall, context),
			If iff => EvaluateIf(iff, context),
			For f => EvaluateFor(f, context),
			Return r => EvaluateReturn(r, context),
			To t => EvaluateTo(t, context),
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

	public class ExpressionNotSupported(Expression expr, ExecutionContext context)
		: ExecutionFailed(context.Type, expr.GetType().Name);

	private static ValueInstance EvaluateVariable(string name, ExecutionContext context) =>
		context.Find(name) ?? (name == Type.ValueLowercase
			? context.This
			: null) ?? throw new VariableNotFound(name, context.Type, context.This);

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
					throw new TestFailed(body.Method, e, last, GetTestFailureDetails(e, ctx));
			}
			if (runOnlyTests && last.Value == null && body.Method.Name != Base.Run &&
				body.Expressions.Count > 1)
				throw new MethodRequiresTest(body.Method, body);
			return runOnlyTests || last.ReturnType.IsError || body.Method.ReturnType == last.ReturnType
				? last
				: body.Method.ReturnType.Name == Base.Character && last.ReturnType.Name == Base.Number
					? new ValueInstance(body.Method.ReturnType, last.Value)
					: body.Method.ReturnType.IsMutable &&
					ctx.This?.ReturnType is GenericTypeImplementation { Generic.Name: Base.Dictionary } &&
					last.ReturnType is GenericTypeImplementation { Generic.Name: Base.List }
						or GenericTypeImplementation
						{
							Generic.Name: Base.Mutable,
							ImplementationTypes:
							[
								GenericTypeImplementation { Generic.Name: Base.List }
							]
						}
						? new ValueInstance(body.Method.ReturnType, ctx.This.Value)
						// If the method requires a mutable return type and the result so far is not, make it!
						: body.Method.ReturnType.IsMutable && !last.ReturnType.IsMutable && last.ReturnType ==
						((GenericTypeImplementation)body.Method.ReturnType).ImplementationTypes[0]
							? new ValueInstance(body.Method.ReturnType, last.Value)
							: throw new ReturnTypeMustMatchMethod(body, last);
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

	/// <summary>
	/// An inline test is typically a standalone boolean expression at the method top level,
	/// not part of a control flow or assignment expression.
	/// </summary>
	private static bool IsStandaloneInlineTest(Expression e) =>
		e.ReturnType.Name == Base.Boolean && e is not If && e is not Return && e is not Declaration &&
		e is not MutableReassignment;

	private string GetTestFailureDetails(Expression expression, ExecutionContext ctx) =>
		expression is Binary { Method.Name: BinaryOperator.Is, Instance: not null } binary &&
		binary.Arguments.Count == 1
			? GetBinaryComparisonDetails(binary, ctx, BinaryOperator.Is)
			: expression is Not { Instance: Binary { Method.Name: BinaryOperator.Is } notBinary } &&
			notBinary.Arguments.Count == 1
				? GetBinaryComparisonDetails(notBinary, ctx, "is not")
				: string.Empty;

	private string GetBinaryComparisonDetails(Binary binary, ExecutionContext ctx, string op) =>
		RunExpression(binary.Instance!, ctx) + " " + op + " " +
		RunExpression(binary.Arguments[0], ctx);

	private ValueInstance EvaluateAndAssign(string name, Expression value, ExecutionContext ctx) =>
		ctx.Set(name, RunExpression(value, ctx));

	private ValueInstance EvaluateReturn(Return r, ExecutionContext ctx) =>
		throw new ReturnSignal(RunExpression(r.Value, ctx));

	private ValueInstance EvaluateIf(If iff, ExecutionContext ctx) =>
		ToBool(RunExpression(iff.Condition, ctx))
			? RunExpression(iff.Then, ctx)
			: iff.OptionalElse != null
				? RunExpression(iff.OptionalElse, ctx)
				: None();

	private ValueInstance EvaluateFor(For f, ExecutionContext ctx)
	{
		var iterator = RunExpression(f.Iterator, ctx);
		var results = new List<ValueInstance>();
		var itemType = GetForValueType(iterator);
		//TODO: this is a bit strange and a regression, this worked fine before and didn't need extra handling, sure we need reverse loop support, but this is 3 loops doing the same thing and way too many parameters ..
		if (iterator.ReturnType.Name == Base.Range &&
			iterator.Value is IDictionary<string, object?> rangeValues &&
			rangeValues.TryGetValue("Start", out var startValue) &&
			rangeValues.TryGetValue("ExclusiveEnd", out var endValue))
		{
			var start = Convert.ToInt32(startValue);
			var end = Convert.ToInt32(endValue);
			if (start <= end)
				for (var index = start; index < end; index++)
					ExecuteForIteration(f, ctx, iterator, results, itemType, index);
			else
				for (var index = start; index > end; index--)
					ExecuteForIteration(f, ctx, iterator, results, itemType, index);
		}
		else
		{
			var loopRange = iterator.ReturnType.Name == Base.Range
				? iterator.GetRange()
				: new Range(0, iterator.GetIteratorLength());
			for (var index = loopRange.Start.Value; index < loopRange.End.Value; index++)
				ExecuteForIteration(f, ctx, iterator, results, itemType, index);
		}
		return ShouldConsolidateForResult(results, ctx) ?? new ValueInstance(results.Count == 0
			? iterator.ReturnType
			: genericListType.GetGenericImplementation(results[0].ReturnType), results);
	}

	private void ExecuteForIteration(For f, ExecutionContext ctx, ValueInstance iterator,
		ICollection<ValueInstance> results, Type itemType, int index)
	{
		var loopContext =
			new ExecutionContext(ctx.Type, ctx.Method) { This = ctx.This, Parent = ctx };
		loopContext.Set(Type.IndexLowercase, new ValueInstance(numberType, index));
		var value = iterator.GetIteratorValue(index);
		loopContext.Set(Type.ValueLowercase,
			value as ValueInstance ?? new ValueInstance(itemType, value));
		foreach (var customVariable in f.CustomVariables)
			if (customVariable is VariableCall variableCall)
				loopContext.Set(variableCall.Variable.Name,
					new ValueInstance(variableCall.ReturnType, value));
		//TODO: if this is a return, for loop should be aborted, we found the result! TODO: we need a test for this first!
		var itemResult = RunExpression(f.Body, loopContext);
		// If there was no result (if did not evaluate), no need to add anything
		if (itemResult.ReturnType.Name != Base.None)
			results.Add(itemResult);
	}

	private static ValueInstance? ShouldConsolidateForResult(List<ValueInstance> results,
		ExecutionContext ctx)
	{
		if (ctx.Method.ReturnType.Name == Base.Number)
			return new ValueInstance(ctx.Method.ReturnType,
				results.Sum(value => EqualsExtensions.NumberToDouble(value.Value)));
		if (ctx.Method.ReturnType.Name == Base.Text)
		{
			var text = "";
			foreach (var value in results)
				text += value.ReturnType.Name switch
				{
					Base.Number or Base.Character => (int)EqualsExtensions.NumberToDouble(value.Value),
					Base.Text => ((string)value.Value!)[0],
					_ => throw new NotSupportedException("Can't append to text: " + value)
				};
			return new ValueInstance(ctx.Method.ReturnType, text);
		}
		if (ctx.Method.ReturnType.Name == Base.Boolean)
			return new ValueInstance(ctx.Method.ReturnType, results.Any(value => value.Value is true));
		return null;
	}

	private Type GetForValueType(ValueInstance iterator) =>
		iterator.ReturnType is GenericTypeImplementation { Generic.Name: Base.List } list
			? list.ImplementationTypes[0]
			: numberType;

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
			return new ValueInstance(to.ConversionType, EqualsExtensions.NumberToDouble(left));
		return !to.Method.IsTrait
			? EvaluateMethodCall(to, ctx)
			: throw new NotSupportedException("Conversion to " + to.ConversionType.Name +
				" not supported");
	}

	private ValueInstance EvaluateNot(Not not, ExecutionContext ctx) =>
		Bool(!ToBool(RunExpression(not.Instance!, ctx)));

	private ValueInstance EvaluateMember(MemberCall member, ExecutionContext ctx) =>
		ctx.This == null && ctx.Type.Members.Contains(member.Member)
			? throw new UnableToCallMemberWithoutInstance(member, ctx)
			: ctx.This?.Value is Dictionary<string, object?> dict &&
			dict.TryGetValue(member.Member.Name, out var value)
				? new ValueInstance(member.ReturnType, value)
				: member.Member.InitialValue != null && member.IsConstant
					? RunExpression(member.Member.InitialValue, ctx)
					: member.Instance is VariableCall { Variable.Name: Type.OuterLowercase }
						? ctx.Parent!.Get(member.Member.Name)
						: ctx.Get(member.Member.Name);

	private ValueInstance EvaluateListCall(ListCall call, ExecutionContext ctx)
	{
		var listInstance = RunExpression(call.List, ctx);
		var indexValue = RunExpression(call.Index, ctx);
		var index = Convert.ToInt32(EqualsExtensions.NumberToDouble(indexValue.Value));
		if (listInstance.Value is IList list)
			return list[index] as ValueInstance ?? new ValueInstance(call.ReturnType, list[index]);
		if (listInstance.Value is IDictionary<string, object?> members &&
			(members.TryGetValue("Elements", out var elements) ||
				members.TryGetValue("elements", out elements)) && elements is IList memberList)
			return memberList[index] as ValueInstance ??
				new ValueInstance(call.ReturnType, memberList[index]);
		if (listInstance.Value is string text)
			return new ValueInstance(call.ReturnType, (int)text[index]);
		throw new InvalidOperationException("List call can only be used on iterators, got: " +
			listInstance); //TODO: proper exception
	}

	public class UnableToCallMemberWithoutInstance(MemberCall member, ExecutionContext ctx)
		: Exception(member + ", context " + ctx);

	private ValueInstance EvaluateMethodCall(MethodCall call, ExecutionContext ctx)
	{
		var op = call.Method.Name;
		if (IsArithmetic(op) || IsCompare(op) || IsLogical(op))
			return EvaluateArithmeticOrCompareOrLogical(call, ctx);
		// If the instance is an expression (like 'not true'), evaluate that, otherwise use context
		var instance = call.Instance != null
			? RunExpression(call.Instance, ctx)
			: call.Method.Name != Method.From
				? ctx.This
				: null;
		var args = new List<ValueInstance>(call.Arguments.Count);
   foreach (var a in call.Arguments)
			args.Add(RunExpression(a, ctx));

		// Special-case: mutate Dictionary instance members when calling Add so the runtime
		// representation (a List of ValueInstance elements) is updated with a ValueInstance
		// wrapping the provided pair. This avoids type-mismatches when the argument value is
		// an IList<ValueInstance> (the tuple) and the underlying list expects ValueInstance.
		if (instance != null && instance.ReturnType is GenericTypeImplementation { Generic.Name: Base.Dictionary }
			&& instance.Value is IDictionary<string, object?> members && args.Count > 0 && call.Method.Name == "Add")
		{
			var listMemberName = instance.ReturnType.Members.FirstOrDefault(member =>
				member.Type is GenericTypeImplementation { Generic.Name: Base.List } ||
				member.Type.Name == Base.List)?.Name ?? Type.ElementsLowercase;
			if (members.TryGetValue(listMemberName, out var listObj) && listObj is IList list)
			{
       // element type is the pair List(...) inside the dictionary member list
				var listMemberType = instance.ReturnType.Members.FirstOrDefault(member =>
					member.Type is GenericTypeImplementation { Generic.Name: Base.List } ||
					member.Type.Name == Base.List)?.Type ?? instance.ReturnType.GetType(Base.List);
				var elementType = listMemberType is GenericTypeImplementation { Generic.Name: Base.List } listType
					? listType.ImplementationTypes[0]
					: listMemberType;
				if (args.Count == 2)
				{
					list.Add(new ValueInstance(elementType, new List<ValueInstance> { args[0], args[1] }));
					return instance;
				}
				var pairArg = args[0];
				if (pairArg.ReturnType.IsSameOrCanBeUsedAs(elementType))
				{
					list.Add(pairArg);
					return instance;
				}
				list.Add(new ValueInstance(elementType, pairArg.Value));
				return instance;
			}
		}
		return Execute(call.Method, instance, args, ctx);
	}

	private static bool IsArithmetic(string name) =>
		name is BinaryOperator.Plus or BinaryOperator.Minus or BinaryOperator.Multiply
			or BinaryOperator.Divide or BinaryOperator.Modulate or BinaryOperator.Power;

	private static bool IsCompare(string name) =>
		name is BinaryOperator.Greater or BinaryOperator.Smaller or BinaryOperator.Is
			or BinaryOperator.GreaterOrEqual or BinaryOperator.SmallerOrEqual or UnaryOperator.Not;

	private static bool IsLogical(string name) =>
		name is BinaryOperator.And or BinaryOperator.Or or BinaryOperator.Xor or UnaryOperator.Not;

	private ValueInstance EvaluateArithmeticOrCompareOrLogical(MethodCall call,
		ExecutionContext ctx)
	{
		if (call.Instance == null || call.Arguments.Count != 1)
			throw new InvalidOperationException(
				"Binary call must have instance and 1 argument"); //ncrunch: no coverage
		var leftInstance = RunExpression(call.Instance, ctx);
		var rightInstance = RunExpression(call.Arguments[0], ctx);
		return IsArithmetic(call.Method.Name)
			? ExecuteArithmeticOperation(call, ctx, leftInstance, rightInstance)
			: IsCompare(call.Method.Name)
				? ExecuteComparisonOperation(call, ctx, leftInstance, rightInstance)
				: ExecuteBinaryOperation(call, ctx, leftInstance, rightInstance);
	}

	private ValueInstance ExecuteArithmeticOperation(MethodCall call, ExecutionContext ctx,
		ValueInstance leftInstance, ValueInstance rightInstance)
	{
		var op = call.Method.Name;
		var left = leftInstance.Value;
		var right = rightInstance.Value;
		//TODO: these are just shortcuts for Number operators, but we don't actually execute them
		//TODO: in any case, for non datatypes we need to call the operators, only allow this for Boolean, Number, Text, rest should be called!
		if (leftInstance.ReturnType.Name == Base.Number &&
			rightInstance.ReturnType.Name == Base.Number)
		{
			var l = EqualsExtensions.NumberToDouble(left);
			var r = EqualsExtensions.NumberToDouble(right);
			return op switch
			{
				BinaryOperator.Plus => Number(l + r),
				BinaryOperator.Minus => Number(l - r),
				BinaryOperator.Multiply => Number(l * r),
				BinaryOperator.Divide => Number(l / r),
				BinaryOperator.Modulate => Number(l % r),
				BinaryOperator.Power => Number(Math.Pow(l, r)),
				_ => ExecuteMethodCall(call, leftInstance, ctx)
			};
		}
		if (leftInstance.ReturnType.Name == Base.Text && rightInstance.ReturnType.Name == Base.Text)
		{
			return op == BinaryOperator.Plus
				? new ValueInstance(leftInstance.ReturnType, (string)left! + (string)right!)
				: throw new NotSupportedException("Only + operator is supported for Text, got: " + op);
		}
		if (leftInstance.ReturnType.IsIterator && rightInstance.ReturnType.IsIterator)
		{
			if (left is not IList<ValueInstance> leftList ||
				right is not IList<ValueInstance> rightList)
				throw new InvalidOperationException(
					"Expected List<ValueInstance> for iterator operation, " +
					"other iterators are not yet supported: left=" + left + ", right=" + right);
			if (op is BinaryOperator.Multiply or BinaryOperator.Divide &&
				leftList.Count != rightList.Count)
				return Error(ListsHaveDifferentDimensions, ctx, call);
			return op switch
			{
				BinaryOperator.Plus => CombineLists(leftInstance.ReturnType, leftList, rightList),
				BinaryOperator.Minus => SubtractLists(leftInstance.ReturnType, leftList, rightList),
				BinaryOperator.Multiply => MultiplyLists(leftInstance.ReturnType, leftList, rightList),
				BinaryOperator.Divide => DivideLists(leftInstance.ReturnType, leftList, rightList),
				_ => throw new NotSupportedException(
					"Only +, -, *, / operators are supported for Lists, got: " + op)
			};
		}
		if (leftInstance.ReturnType.IsIterator && rightInstance.ReturnType.Name == Base.Number)
		{
			if (left is not IList<ValueInstance> leftList)
				throw new InvalidOperationException("Expected left list for iterator operation " + op +
					": left=" + left + ", right=" + right);
			if (op == BinaryOperator.Plus)
				return AddToList(leftInstance.ReturnType, leftList, rightInstance);
			if (op == BinaryOperator.Minus)
				return RemoveFromList(leftInstance.ReturnType, leftList, rightInstance);
			if (right is not double rightNumber)
				throw new InvalidOperationException("Expected right number for iterator operation " + op +
					": left=" + left + ", right=" + right);
			if (op == BinaryOperator.Multiply)
				return MultiplyList(leftInstance.ReturnType, leftList, rightNumber);
			if (op == BinaryOperator.Divide)
				return DivideList(leftInstance.ReturnType, leftList, rightNumber);
			throw new NotSupportedException(
				"Only +, -, *, / operators are supported for List and Number, got: " + op);
		}
		return ExecuteMethodCall(call, leftInstance, ctx);
	}

	private ValueInstance ExecuteComparisonOperation(MethodCall call, ExecutionContext ctx,
		ValueInstance leftInstance, ValueInstance rightInstance)
	{
		var op = call.Method.Name;
		var left = leftInstance.Value;
		var right = rightInstance.Value;
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
				return op is BinaryOperator.Is
					? Bool(matches)
					: Bool(!matches);
			}
			//TODO: support conversions needed for Character, maybe Number <-> Text
			if (call.Instance!.ReturnType.Name == Base.Character && right is string rightText)
			{
				right = (int)rightText[0];
				rightInstance = new ValueInstance(call.Instance.ReturnType, right);
			}
			if (call.Instance.ReturnType.Name == Base.Text && right is int rightInt)
			{
				right = rightInt + "";
				rightInstance = new ValueInstance(call.Instance.ReturnType, right);
			}
			var equals = leftInstance.Equals(rightInstance);
			return op is BinaryOperator.Is
				? Bool(equals)
				: Bool(!equals);
		}
		var l = EqualsExtensions.NumberToDouble(left);
		var r = EqualsExtensions.NumberToDouble(right);
		return op switch
		{
			BinaryOperator.Greater => Bool(l > r),
			BinaryOperator.Smaller => Bool(l < r),
			BinaryOperator.GreaterOrEqual => Bool(l >= r),
			BinaryOperator.SmallerOrEqual => Bool(l <= r),
			_ => ExecuteMethodCall(call, leftInstance, ctx)
		};
	}

	private ValueInstance ExecuteBinaryOperation(MethodCall call, ExecutionContext ctx,
		ValueInstance leftInstance, ValueInstance rightInstance)
	{
		var left = leftInstance.Value;
		var right = rightInstance.Value;
		return call.Method.Name switch
		{
			BinaryOperator.And => Bool(ToBool(left) && ToBool(right)),
			BinaryOperator.Or => Bool(ToBool(left) || ToBool(right)),
			BinaryOperator.Xor => Bool(ToBool(left) ^ ToBool(right)),
			_ => ExecuteMethodCall(call, leftInstance, ctx)
		};
	}

	private static ValueInstance CombineLists(Type listType, ICollection<ValueInstance> leftList,
		ICollection<ValueInstance> rightList)
	{
		var combined = new List<ValueInstance>(leftList.Count + rightList.Count);
		var isLeftText = listType is GenericTypeImplementation { Generic.Name: Base.List } list &&
			list.ImplementationTypes[0].Name == Base.Text;
		foreach (var item in leftList)
			combined.Add(item);
		foreach (var item in rightList)
			combined.Add(isLeftText && item.ReturnType.Name != Base.Text
				? new ValueInstance(listType.GetType(Base.Text), item.Value?.ToString())
				: item);
		return new ValueInstance(listType, combined);
	}

	private static ValueInstance SubtractLists(Type listType, IEnumerable<ValueInstance> leftList,
		IEnumerable<ValueInstance> rightList)
	{
		var remainder = new List<ValueInstance>();
		foreach (var item in leftList)
			remainder.Add(item);
		foreach (var item in rightList)
			remainder.Remove(item);
		return new ValueInstance(listType, remainder);
	}

	private static ValueInstance MultiplyLists(Type leftListType, IList<ValueInstance> leftList,
		IList<ValueInstance> rightList)
	{
		var result = new List<ValueInstance>();
		for (var index = 0; index < leftList.Count; index++)
			result.Add(new ValueInstance(leftListType.GetType(Base.Number),
				EqualsExtensions.NumberToDouble(leftList[index].Value) *
				EqualsExtensions.NumberToDouble(rightList[index].Value)));
		return new ValueInstance(leftListType, result);
	}

	private static ValueInstance DivideLists(Type leftListType, IList<ValueInstance> leftList,
		IList<ValueInstance> rightList)
	{
		var result = new List<ValueInstance>();
		for (var index = 0; index < leftList.Count; index++)
			result.Add(new ValueInstance(leftListType.GetType(Base.Number),
				EqualsExtensions.NumberToDouble(leftList[index].Value) /
				EqualsExtensions.NumberToDouble(rightList[index].Value)));
		return new ValueInstance(leftListType, result);
	}

	private static ValueInstance AddToList(Type leftListType, ICollection<ValueInstance> leftList,
		ValueInstance right)
	{
		var combined = new List<ValueInstance>(leftList.Count + 1);
		var isLeftText = leftListType is GenericTypeImplementation { Generic.Name: Base.List } list &&
			list.ImplementationTypes[0].Name == Base.Text;
		foreach (var item in leftList)
			combined.Add(item);
		combined.Add(isLeftText && right.ReturnType.Name != Base.Text
			? new ValueInstance(leftListType.GetType(Base.Text), ConvertToText(right.Value))
			: right);
		return new ValueInstance(leftListType, combined);
	}

	private static string ConvertToText(object? value) =>
		value switch
		{
			string text => text,
			double number => number.ToString(CultureInfo.InvariantCulture),
			int number => number.ToString(CultureInfo.InvariantCulture),
			_ => value?.ToString() ?? string.Empty
		};

	private static ValueInstance RemoveFromList(Type leftListType,
		IEnumerable<ValueInstance> leftList, ValueInstance right)
	{
		var result = new List<ValueInstance>();
		foreach (var item in leftList)
			if (!item.Equals(right))
				result.Add(item);
		return new ValueInstance(leftListType, result);
	}

	private static ValueInstance MultiplyList(Type leftListType,
		ICollection<ValueInstance> leftList, double rightNumber)
	{
		var result = new List<ValueInstance>(leftList.Count);
		foreach (var item in leftList)
			result.Add(new ValueInstance(item.ReturnType,
				EqualsExtensions.NumberToDouble(item.Value) * rightNumber));
		return new ValueInstance(leftListType, result);
	}

	private static ValueInstance DivideList(Type leftListType, ICollection<ValueInstance> leftList,
		double rightNumber)
	{
		var result = new List<object?>(leftList.Count);
		foreach (var item in leftList)
			result.Add(new ValueInstance(item.ReturnType,
				EqualsExtensions.NumberToDouble(item.Value) / rightNumber));
		return new ValueInstance(leftListType, result);
	}

	private ValueInstance ExecuteMethodCall(MethodCall call, ValueInstance instance,
		ExecutionContext ctx)
	{
		var args = new List<ValueInstance>(call.Arguments.Count);
		foreach (var a in call.Arguments)
			args.Add(RunExpression(a, ctx));
		return Execute(call.Method, instance, args, ctx);
	}

	private ValueInstance None() => new(noneType, null);
	private readonly Type noneType = basePackage.FindType(Base.None)!;
	private ValueInstance Bool(bool b) => new(boolType, b);
	private readonly Type boolType = basePackage.FindType(Base.Boolean)!;
	private readonly Type genericListType = basePackage.FindType(Base.List)!;
	private readonly Type errorType = basePackage.FindType(Base.Error)!;
	private readonly Type stacktraceType = basePackage.FindType(Base.Stacktrace)!;
	private readonly Type methodType = basePackage.FindType(Base.Method)!;
	private readonly Type typeType = basePackage.FindType(Base.Type)!;
	public const string ListsHaveDifferentDimensions = "listsHaveDifferentDimensions";

	private ValueInstance Error(string name, ExecutionContext ctx, Expression? source = null)
	{
		var stacktraceList = new List<object?> { CreateStacktrace(ctx, source) };
		var errorMembers = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
		foreach (var member in errorType.Members)
			errorMembers[member.Name] = member.Type.Name switch
			{
				Base.Name or Base.Text => name,
				_ when member.Type.Name == Base.List || member.Type is GenericTypeImplementation
				{
					Generic.Name: Base.List
				} => stacktraceList,
				_ => throw new NotSupportedException("Error member not supported: " + member)
			};
		return new ValueInstance(errorType, errorMembers);
	}

	private Dictionary<string, object?> CreateStacktrace(ExecutionContext ctx, Expression? source)
	{
		var members = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
		foreach (var member in stacktraceType.Members)
			members[member.Name] = member.Type.Name switch
			{
				Base.Method => CreateMethodValue(ctx.Method),
				Base.Text or Base.Name => ctx.Method.Type.FilePath,
				Base.Number => (double)(source?.LineNumber ?? ctx.Method.TypeLineNumber),
				_ => throw new NotSupportedException("Stacktrace member not supported: " + member)
			};
		return members;
	}

	private Dictionary<string, object?> CreateMethodValue(Method method)
	{
		var values = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
		foreach (var member in methodType.Members)
			values[member.Name] = member.Type.Name switch
			{
				Base.Name or Base.Text => method.Name,
				Base.Type => CreateTypeValue(method.Type),
				_ => throw new NotSupportedException("Method member not supported: " + member)
			};
		return values;
	}

	private Dictionary<string, object?> CreateTypeValue(Type type)
	{
		var values = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
		foreach (var member in typeType.Members)
			values[member.Name] = member.Type.Name switch
			{
				Base.Name => type.Name,
				Base.Text => type.Package.FullName,
				_ => throw new NotSupportedException("Type member not supported: " + member)
			};
		return values;
	}

	private static bool ToBool(object? v) =>
		v switch
		{
			bool b => b,
			ValueInstance vi => ToBool(vi.Value),
			Value { ReturnType.Name: Base.Boolean, Data: bool bv } => bv,
			_ => throw new InvalidOperationException("Expected Boolean, got: " + v)
		};

	public sealed class ComparisonsToNullAreNotAllowed(Method method, object? left, object? right)
		: ExecutionFailed(method, $"{left} is {right}");

	private ValueInstance Number(double n) => new(numberType, n);
	private readonly Type numberType = basePackage.FindType(Base.Number)!;

	private sealed class ReturnSignal(ValueInstance value) : Exception
	{
		public ValueInstance Value { get; } = value;
	}
}