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

	private bool InitializesMembers(Method method) =>
		memberInitializingFroms.GetOrAdd(method, static from =>
			from.lines.Skip(1).Any(line => from.Type.Members.Any(member =>
				line.StartsWith("\t" + member.Name + " = ", StringComparison.Ordinal))));

	private readonly ConcurrentDictionary<Method, bool> memberInitializingFroms = new();

	/// <summary>
	/// A custom from(..) assigns members, start from default member values and run its body.
	/// </summary>
	private ValueInstance ExecuteMemberInitializingFrom(Method method, ValueInstance[] args,
		ExecutionContext? parentContext)
	{
		Statistics.FromCreationsCount++;
		var members = method.Type.Members;
		var values = new ValueInstance[members.Count];
		for (var memberIndex = 0; memberIndex < members.Count; memberIndex++)
			values[memberIndex] = CreateDefaultMemberValue(members[memberIndex].Type);
		var newInstance = new ValueInstance(method.Type, values);
		var context = CreateExecutionContext(method, newInstance, args, parentContext, false);
		try
		{
			RunExpression(method.GetBodyAndParseIfNeeded(), context);
		}
		finally
		{
			DisposeTrackedValues(context);
			ReturnContext(context);
		}
		ValidateMemberConstraints(method, newInstance.TryGetValueTypeInstance()!.Values);
		return newInstance;
	}

	private ValueInstance GetFromConstructorValue(Method method, IReadOnlyList<ValueInstance> args)
	{
		Statistics.FromCreationsCount++;
		if (args.Count == 0 && method.Type.IsText)
			return new ValueInstance("");
		if (args.Count == 0 && (method.Type.IsCharacter || method.Type.IsNumber))
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
			if (!args[index].IsSameOrCanBeUsedAs(parameter.Type) && !parameter.Type.IsIterator &&
				!IsSingleCharacterTextArgument(parameter.Type, args[index]))
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
		ValidateMemberConstraints(method, values);
		if (!method.Type.IsMutable && values.Length == 1 &&
			values[0].IsSameOrCanBeUsedAs(method.Type))
			return values[0];
		TryPreFillConstrainedListMembers(method.Type, values, method);
		return new ValueInstance(method.Type, values);
	}

	private void ValidateMemberConstraints(Method method, ValueInstance[] values)
	{
		var members = method.Type.Members;
		for (var memberIndex = 0; memberIndex < members.Count; memberIndex++)
			if (members[memberIndex].Constraints is { } constraints && values[memberIndex].HasValue &&
				!members[memberIndex].Type.IsList)
				foreach (var constraint in constraints)
					if (!EvaluateConstraint(method, values, memberIndex, constraint))
						throw new InterpreterExecutionFailed(method, "Constraint " + constraint + " of member " +
							members[memberIndex].Name + " failed for value " + values[memberIndex]);
	}

	private bool EvaluateConstraint(Method method, ValueInstance[] values, int memberIndex,
		Expression constraint)
	{
		var members = method.Type.Members;
		var context = RentContext(members[memberIndex].Type, method, values[memberIndex], null);
		try
		{
			for (var index = 0; index < members.Count; index++)
				if (values[index].HasValue)
					context.Variables[members[index].Name] = values[index];
			return RunExpression(constraint, context).Boolean;
		}
		finally
		{
			ReturnContext(context);
		}
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
			if (member.IsConstant)
			{
				values[memberIndex] = noneInstance;
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

	private ValueInstance? TryAutoCreateInstance(Type type, HashSet<string>? creating = null)
	{
		creating ??= [];
		if (type.IsText)
			return new ValueInstance("");
		if (type.IsNumber)
			return new ValueInstance(numberType, 0);
		if (type.IsBoolean)
			return new ValueInstance(booleanType, false);
		if (type.IsCharacter)
			return new ValueInstance(characterType, 0);
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
}
