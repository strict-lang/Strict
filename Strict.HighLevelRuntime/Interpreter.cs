using System.Collections.Concurrent;
using Strict.Expressions;
using Strict.Language;
using System.Runtime.CompilerServices;
using Type = Strict.Language.Type;

[assembly: InternalsVisibleTo("Strict.HighLevelRuntime.Tests")]
[assembly: InternalsVisibleTo("Strict.TestRunner")]

namespace Strict.HighLevelRuntime;

public class Interpreter
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
	}

	private readonly TestBehavior behavior;
	internal readonly Type noneType;
	internal readonly ValueInstance noneInstance;
	internal readonly Type booleanType;
	internal readonly ValueInstance trueInstance;
	internal readonly ValueInstance falseInstance;
	internal readonly Type numberType;
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

	private ValueInstance CreateFullInstance(Type type)
	{
		var members = type.Members;
		if (members.Count == 0)
			return noneInstance; //ncrunch: no coverage
		var values = new ValueInstance[members.Count];
		for (var i = 0; i < members.Count; i++)
		{
			var autoValue = TryAutoCreateInstance(members[i].Type);
			values[i] = autoValue ?? GetDefaultValue(members[i].Type);
		}
		return new ValueInstance(type, values);
	}

	public class MethodNotFound(Type type, string methodName) : InterpreterExecutionFailed(type, methodName);

	private ValueInstance GetDefaultValue(Type type)
	{
		if (type.IsNumber)
			return new ValueInstance(numberType, 0);
		//ncrunch: no coverage start
		if (type.IsText)
			return new ValueInstance("");
		return type.IsBoolean
			? new ValueInstance(booleanType, false)
			: noneInstance;
	} //ncrunch: no coverage end

	public ValueInstance Execute(Method method, ValueInstance instance,
		ValueInstance[] args, ExecutionContext? parentContext = null, bool runOnlyTests = false)
	{
		Statistics.MethodCount++;
    args = NormalizeArguments(method, args, parentContext);
		ValidateInstanceAndArguments(method, instance, args, parentContext);
		if (method is { Name: Method.From, Type.IsGeneric: false })
			return instance.Equals(noneInstance)
				? GetFromConstructorValue(method, args)
				: throw new MethodCall.CannotCallFromConstructorWithExistingInstance();
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
				if (ShouldIgnoreGenericListTestParseFailure(method, inner) ||
					IsKnownParserLimitation(inner))
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
			return context.ExitMethodAndReturnValue ??
				result.ApplyMethodReturnTypeMutable(method.ReturnType);
		}
		finally
		{
			ReturnContext(context);
		}
	}

	private bool TryExecuteNativeFileMethod(Method method, ValueInstance instance,
		IReadOnlyList<ValueInstance> args, out ValueInstance result)
	{
		result = noneInstance;
		if (method.Type.Name != Type.File)
			return false;
		var path = GetFilePath(instance, method);
		switch (method.Name)
		{
		case "ReadText":
			result = new ValueInstance(File.ReadAllText(path));
			return true;
		case "ReadBytes":
			result = CreateBytesValue(method, File.ReadAllBytes(path));
			return true;
		case "Write":
			WriteFile(path, args, method);
			return true;
		case "Delete":
			File.Delete(path);
			return true;
		case "Exists":
			result = ToBoolean(File.Exists(path));
			return true;
		case "Length":
			result = new ValueInstance(numberType, new FileInfo(path).Length);
			return true;
		default:
			return false;
		}
	}

	private static string GetFilePath(ValueInstance instance, Method method)
	{
		if (!FileValue.TryGetPath(instance, out var path))
			throw new InterpreterExecutionFailed(method, "File instance has no path Text member");
		return path;
	}

	private static void WriteFile(string path, IReadOnlyList<ValueInstance> args, Method method)
	{
		if (args.Count == 0)
			throw new MissingArgument(method, "text", args);
		if (args[0].IsText)
			File.WriteAllText(path, args[0].Text);
		else if (args[0].IsList)
			File.WriteAllBytes(path, FileValue.GetBytes(args[0]));
		else
			throw new InvalidTypeForArgument(method.Type, args, 0);
	}

	private ValueInstance CreateBytesValue(Method method, byte[] bytes)
	{
		var byteType = method.GetType(Type.Byte);
		var bytesType = method.GetListImplementationType(byteType);
		return FileValue.CreateBytes(bytesType, byteType, bytes);
	}

	private static bool ShouldIgnoreGenericListTestParseFailure(Method method, Exception inner) =>
		method.Type.IsGeneric && method.Type.Name == Type.List &&
		inner is Type.GenericTypesCannotBeUsedDirectlyUseImplementation;

	private static bool IsKnownParserLimitation(Exception inner) =>
		inner is ParsingFailed &&
		(inner.InnerException is Type.NoMatchingMethodFound
				or Type.ArgumentsDoNotMatchMethodParameters ||
			inner.Message.Contains("Use number iteration"));

	private static bool ShouldSkipGenericListTestValidation(Method method, bool runOnlyTests) =>
		runOnlyTests && method.Type is { IsGeneric: true, Name: Type.List or Type.Dictionary };

	private static bool ShouldSkipKnownStrictBaseMethodValidation(Method method, bool runOnlyTests) =>
		runOnlyTests && (method.Type.IsGeneric && method.Type.Name == Type.List ||
			method.Type.Name == Type.Number &&
			(method.Name == "digits" || method.Name == BinaryOperator.To && method.ReturnType.IsText) ||
			method.Type.IsText && method.Name == "Split" ||
			method.Type.Name == Type.File ||
			method.Type.Name == "StrictFileCompiler");

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
		for (var current = parentContext; current != null; current = current.Parent)
			if (current.Method == method &&
				AreSameInstanceForRecursionCheck(current.This, instance) &&
				DoArgumentsMatch(method, args, current.Variables))
				throw new StackOverflowCallingItselfWithSameInstanceAndArguments(method, instance, args,
					current);
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
		ValueInstance? instance, IReadOnlyList<ValueInstance> args, ExecutionContext parentContext)
		: InterpreterExecutionFailed(method, "Stack overflow detected while calling " +
			FormatCall(method, instance, args) + ". Matching parent call chain: " +
			FormatParentChain(parentContext))
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

	private ValueInstance GetFromConstructorValue(Method method, IReadOnlyList<ValueInstance> args)
	{
		Statistics.FromCreationsCount++;
		if (args.Count == 0 && method.Type.IsText)
			return new ValueInstance("");
		if (args.Count == 0 && method.Type.IsCharacter)
			return new ValueInstance(method.Type, 0);
		if ((method.Type.IsCharacter || method.Type.IsNumber || method.Type.IsEnum) && args.Count == 1)
		{
			if (IsSingleCharacterTextArgument(method.Type, args[0]))
				return new ValueInstance(method.Type, args[0].Text[0]);
			if (!args[0].IsText || args[0].IsSameOrCanBeUsedAs(method.Type))
				return new ValueInstance(method.Type, args[0].Number);
		}
		if (method.Type.IsList)
			return new ValueInstance(method.Type, args.ToArray());
		if (method.Type.IsDictionary)
			return args[0].IsDictionary
				? args[0]
				: new ValueInstance(method.Type, FillDictionaryFromListKeyAndValues(args[0]));
		var typeMembers = method.Type.Members;
		if (typeMembers.Count == 0)
			return noneInstance;
		var values = new ValueInstance[typeMembers.Count];
		for (var index = 0; index < args.Count; index++)
		{
			var parameter = method.Parameters[index];
			if (!args[index].IsSameOrCanBeUsedAs(parameter.Type) &&
				!parameter.Type.IsIterator && !IsSingleCharacterTextArgument(parameter.Type, args[index]))
				throw new InvalidTypeForArgument(method.Type, args, index);
			var memberIndex = GetMemberIndexForParameter(typeMembers, parameter, index);
			values[memberIndex] = IsSingleCharacterTextArgument(parameter.Type, args[index])
				? new ValueInstance(characterType, args[index].Text[0])
				: args[index];
		}
		for (var index = args.Count; index < method.Parameters.Count; index++)
		{
			var parameter = method.Parameters[index];
			var memberIndex = GetMemberIndexForParameter(typeMembers, parameter, index);
			if (memberIndex >= typeMembers.Count)
				continue;
			var memberType = typeMembers[memberIndex].Type;
			if (parameter.DefaultValue != null)
			{
				var defaultVal = RunExpression(parameter.DefaultValue,
					RentContext(method.Type, method, noneInstance, null));
				values[memberIndex] = memberType.IsList && !defaultVal.IsSameOrCanBeUsedAs(memberType)
					? new ValueInstance(memberType, Array.Empty<ValueInstance>())
					: defaultVal;
			}
			else
			{
				var autoValue = TryAutoCreateInstance(memberType);
				if (autoValue != null)
					values[memberIndex] = autoValue.Value;
			}
		}
		for (var memberIndex = 0; memberIndex < typeMembers.Count; memberIndex++)
			if (!values[memberIndex].HasValue && typeMembers[memberIndex].Type.IsList)
				values[memberIndex] = new ValueInstance(typeMembers[memberIndex].Type,
					Array.Empty<ValueInstance>());
		TryPreFillConstrainedListMembers(method.Type, values, method);
		return new ValueInstance(method.Type, values);
	}

	private void TryPreFillConstrainedListMembers(Type targetType, ValueInstance[] values,
		Method method)
	{
		var members = targetType.Members;
		for (var memberIndex = 0; memberIndex < members.Count; memberIndex++)
		{
			if (!values[memberIndex].IsList || values[memberIndex].List.Items.Count > 0 ||
				members[memberIndex].Constraints == null)
				continue;
			var constrainedLength = TryGetConstrainedLength(targetType, values, members[memberIndex],
				method);
			if (constrainedLength is not > 0)
				continue;
			var elementType = members[memberIndex].Type is GenericTypeImplementation genericList
				? genericList.ImplementationTypes[0]
				: members[memberIndex].Type;
			var elements = new ValueInstance[constrainedLength.Value];
			for (var elementIndex = 0; elementIndex < constrainedLength.Value; elementIndex++)
				elements[elementIndex] = CreateDefaultMemberValue(elementType);
			values[memberIndex] = new ValueInstance(members[memberIndex].Type, elements);
		}
	}

	private int? TryGetConstrainedLength(Type targetType, ValueInstance[] values, Member member,
		Method method)
	{
		foreach (var constraint in member.Constraints!)
		{
			if (constraint is not Binary { Method.Name: BinaryOperator.Is } binary ||
				binary.Instance?.ToString() != "Length")
				continue;
			if (binary.Arguments[0] is Value numberValue)
				return (int)numberValue.Data.Number;
			return TryEvaluateLengthInMemberScope(targetType, values, binary.Arguments[0], method);
		}
		return null;
	}

	private int? TryEvaluateLengthInMemberScope(Type targetType, ValueInstance[] values,
		Expression lengthExpression, Method method)
	{
		var context = RentContext(targetType, method, noneInstance, null);
		try
		{
			for (var memberIndex = 0; memberIndex < targetType.Members.Count; memberIndex++)
				if (values[memberIndex].HasValue)
					context.Variables[targetType.Members[memberIndex].Name] = values[memberIndex];
			return (int)RunExpression(lengthExpression, context).Number;
		}
		catch
		{
			return null;
		}
		finally
		{
			ReturnContext(context);
		}
	}

	private ValueInstance CreateDefaultMemberValue(Type type)
	{
		if (type.IsText)
			return new ValueInstance("");
		if (type.IsBoolean)
			return new ValueInstance(type, false);
		if (type.IsNumber || type.IsCharacter || type.IsEnum)
			return new ValueInstance(type, 0);
		if (type.IsNone)
			return noneInstance;
		if (type.IsList)
			return new ValueInstance(type, Array.Empty<ValueInstance>());
		if (type.IsDictionary)
			return new ValueInstance(type, new Dictionary<ValueInstance, ValueInstance>());
		var members = type.Members;
		if (members.Count == 0)
			return new ValueInstance(type, 0);
		var values = new ValueInstance[members.Count];
		for (var memberIndex = 0; memberIndex < members.Count; memberIndex++)
		{
			var member = members[memberIndex];
			if (member.Type.IsTrait)
			{
				var traitValue = TryAutoCreateInstance(member.Type);
				values[memberIndex] = traitValue ?? noneInstance;
				continue;
			}
			if (member.InitialValue is Value initialValue)
			{
				values[memberIndex] = initialValue.Data;
				continue;
			}
			if (member.Type.IsList)
			{
				values[memberIndex] = new ValueInstance(member.Type, Array.Empty<ValueInstance>());
				continue;
			}
			values[memberIndex] = CreateDefaultMemberValue(member.Type);
		}
		return new ValueInstance(type, values);
	}

	private static readonly IReadOnlyDictionary<string, string> TraitImplementationRegistry =
		new Dictionary<string, string> { { Type.TextWriter, Type.System } };

	private ValueInstance? TryAutoCreateInstance(Type type, HashSet<string>? creating = null)
	{
		creating ??= [];
		if (type.IsText)
		{
			return new ValueInstance("");
		}
		if (type.IsNumber)
		{
			return new ValueInstance(numberType, 0);
		}
		if (type.IsBoolean)
		{
			return new ValueInstance(booleanType, false);
		}
		if (type.IsCharacter)
		{
			return new ValueInstance(characterType, 0);
		}
		if (type.IsTrait)
		{
			if (!TraitImplementationRegistry.TryGetValue(type.Name, out var concreteName))
				return null; //ncrunch: no coverage
			var concreteType = type.FindType(concreteName);
			return concreteType == null
				? null
				// ReSharper disable once TailRecursiveCall
				: TryAutoCreateInstance(concreteType, creating);
		}
		if (!creating.Add(type.Name))
		{
			var dummyValues = new ValueInstance[type.Members.Count];
			for (var i = 0; i < dummyValues.Length; i++)
				dummyValues[i] = noneInstance;
			return new ValueInstance(type, dummyValues);
		}
		var members = type.Members;
		if (members.Count == 0)
		{
			creating.Remove(type.Name);
			return null;
		}
		var values = new ValueInstance[members.Count];
		for (var i = 0; i < members.Count; i++)
		{
			var memberValue = TryAutoCreateInstance(members[i].Type, creating);
			if (memberValue == null)
			{ //ncrunch: no coverage start
				creating.Remove(type.Name);
				return null;
			} //ncrunch: no coverage end
			values[i] = memberValue.Value;
		}
		creating.Remove(type.Name);
		return new ValueInstance(type, values);
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

	private static int GetMemberIndexForParameter(IReadOnlyList<Member> typeMembers,
		Parameter parameter, int fallbackIndex)
	{
		for (var i = 0; i < typeMembers.Count; i++)
			if (typeMembers[i].Name.Equals(parameter.Name, StringComparison.OrdinalIgnoreCase))
				return i;
		return fallbackIndex; //ncrunch: no coverage
	}

	private static bool IsSingleCharacterTextArgument(Type targetType, ValueInstance value) =>
		value is { IsText: true, Text.Length: 1 } && (targetType.IsNumber || targetType.IsCharacter);

	public sealed class InvalidTypeForArgument(Type type, IReadOnlyList<ValueInstance> args,
		int index) : InterpreterExecutionFailed(type, args[index] + " at index=" + index +
			" does not match type=" + type + " Member=" + type.Members[index]);

	public sealed class CannotCallMethodWithWrongInstance(Method method, ValueInstance instance,
		Type expectedInstanceType)
		: InterpreterExecutionFailed(method, instance.ToString() + " is wrong, expected: " +
			expectedInstanceType);

	public sealed class TooManyArguments(Method method, string argument,
		IReadOnlyList<ValueInstance> args) : InterpreterExecutionFailed(method,
		argument + ", given arguments: " + string.Join(", ", args) + ", method " + method.Name +
		" requires these parameters: " + string.Join(", ", method.Parameters));

	public sealed class ArgumentDoesNotMapToMethodParameters(Method method, string message)
		: InterpreterExecutionFailed(method, message);

	public sealed class MissingArgument(Method method, string paramName,
		IReadOnlyList<ValueInstance> args) : InterpreterExecutionFailed(method,
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

	private ValueInstance EvaluateListExpression(List list, ExecutionContext context)
	{
		var constantData = list.TryGetConstantData();
		if (constantData.HasValue)
			return constantData.Value;
		var count = list.Values.Count;
		var values = new ValueInstance[count];
		for (var i = 0; i < count; i++)
			values[i] = RunExpression(list.Values[i], context);
		return new ValueInstance(list.ReturnType, values);
	}

	public class ExpressionNotSupported(Expression expr, ExecutionContext context)
		: InterpreterExecutionFailed(context.Type, expr.GetType().Name); //ncrunch: no coverage

	private ValueInstance EvaluateVariable(string name, ExecutionContext context)
	{
		Statistics.VariableCallCount++;
		return context.Find(name, Statistics) ?? name switch
		{
			Type.ValueLowercase => context.This,
			Type.OuterLowercase => context.Parent!.Get(Type.ValueLowercase, Statistics),
			_ => null
		} ?? throw new ExecutionContext.VariableNotFound(name, context.Type, context.This);
	}

	public ValueInstance EvaluateMemberCall(MemberCall member, ExecutionContext ctx)
	{
		Statistics.MemberCallCount++;
		if (member.Instance is VariableCall { Variable.Name: Type.OuterLowercase })
			return ctx.Parent!.Get(member.Member.Name, Statistics);
		if (member.Member.InitialValue != null && member.IsConstant)
			return RunExpression(member.Member.InitialValue, ctx);
		var instance = member.Instance != null
			? RunExpression(member.Instance, ctx)
			: ctx.This;
		if (instance == null && ctx.Type.Members.Contains(member.Member))
			throw new UnableToCallMemberWithoutInstance(member, ctx); //ncrunch: no coverage
		if (instance is { IsDictionary: true } &&
			member.Member.Name.Equals(Type.ElementsLowercase, StringComparison.OrdinalIgnoreCase))
		{
			var dictionaryItems = instance.Value.GetDictionaryItems();
			var pairs = new ValueInstance[dictionaryItems.Count];
			var pairType = member.Member.Type is { IsList: true, IsGeneric: true }
				? listType.GetFirstImplementation()
				: member.Member.Type;
			var index = 0;
			foreach (var pair in dictionaryItems)
				pairs[index++] = new ValueInstance(pairType, [pair.Key, pair.Value]);
			return new ValueInstance(member.Member.Type, pairs);
		}
		var typeInstance = instance?.TryGetValueTypeInstance();
		if (typeInstance != null && typeInstance.TryGetValue(member.Member.Name, out var value))
			return value;
		if (instance != null && !member.IsConstant && member.Member.Type.Name != Type.Iterator)
			return new ValueInstance(instance.Value, member.Member.Type);
		return ctx.Get(member.Member.Name, Statistics);
	}

	public class UnableToCallMemberWithoutInstance(MemberCall member, ExecutionContext ctx)
		: Exception(member + ", context " + ctx); //ncrunch: no coverage

	public sealed class ReturnTypeMustMatchMethod(Body body, ValueInstance last) : InterpreterExecutionFailed(
		body.Method, "Return value " + last + " does not match method " + body.Method.Name +
		" ReturnType=" + body.Method.ReturnType);

	private readonly ConcurrentDictionary<Method, bool> simpleMethodCache = new();

	/// <summary>
	/// Skip parsing for trivially simple methods during validation to avoid missing-instance errors.
	/// </summary>
	private bool IsSimpleSingleLineMethod(Method method) =>
		simpleMethodCache.GetOrAdd(method, CheckIsSimpleSingleLineMethod);

	private static bool CheckIsSimpleSingleLineMethod(Method method)
	{
		if (method.lines.Count != 2)
			return false;
		var bodyLine = method.lines[1].Trim();
		var hasMethodCalls = bodyLine.Contains('(') && !bodyLine.StartsWith('(');
		if (hasMethodCalls)
			return false;
		var thenCount = CountThenSeparators(bodyLine);
		var operatorCount = CountOperatorWords(bodyLine);
		return thenCount == 0 && operatorCount <= 1 || thenCount == 1 && operatorCount <= 2 ||
			thenCount == 2 && operatorCount == 0;
	}

	private static int CountOperatorWords(string input)
	{
		var span = input.AsSpan();
		var count = 0;
		while (span.Length > 0)
		{
			var spaceIndex = span.IndexOf(' ');
			var word = spaceIndex < 0
				? span
				: span[..spaceIndex];
			if (word is "and" or "or" or "not" or "is")
				count++;
			if (spaceIndex < 0)
				break;
			span = span[(spaceIndex + 1)..];
		}
		return count;
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

	public class MethodRequiresTest(Method method, string body) : InterpreterExecutionFailed(method,
		body.StartsWith("Test execution failed", StringComparison.Ordinal)
			? body
			: $"Method {method.Parent.FullName}.{method.Name}\n{body}")
	{
		public MethodRequiresTest(Method method, Body body) : this(method,
			body + " ({CountExpressionComplexity(body)} expressions)") { }
	}

	public sealed class TestFailed(Method method,	Expression expression, ValueInstance result,
		string details) : InterpreterExecutionFailed(method,
		$"\"{method.Name}\" method failed: {expression}, result: {result}" + (details.Length > 0
			? $", evaluated: {details}"
			: "") + " in" + Environment.NewLine +
		$"{method.Type.FilePath}:line {expression.LineNumber + 1}");

	private ValueInstance EvaluateMutableListElementAssignment(ListCall target, Expression value,
		ExecutionContext ctx)
	{
		Statistics.MutableUsageCount++;
		var newValue = RunExpression(value, ctx);
		var index = (int)RunExpression(target.Index, ctx).Number;
		var listInstance = RunExpression(target.List, ctx);
		listInstance.List.Items[index] = newValue;
		return newValue;
	}

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
		return ToBoolean(!RunExpression(not.Instance!, ctx).Boolean);
	}

	public ValueInstance ToBoolean(bool isTrue) =>
		isTrue
			? trueInstance
			: falseInstance;
}
