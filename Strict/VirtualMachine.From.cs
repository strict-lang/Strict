using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict;

public sealed partial class VirtualMachine
{
	/// <summary>
	/// Handles From constructor calls like SimpleCalculator(2, 3) by creating a ValueInstance
	/// with evaluated argument values for each non-trait member.
	/// </summary>
	private bool TryHandleFromConstructor(Invoke invoke, Type targetType)
	{
		if (targetType is GenericTypeImplementation)
			return false;
		var members = targetType.Members;
		var hasBinaryMembers = TryGetBinaryMembers(targetType, out var binaryMembers);
		if (members.Count == 0 && hasBinaryMembers)
		{
			//TODO: called almost 1 million times? wtf?
			Memory.Registers[invoke.Register] = new ValueInstance(targetType,
				CreateConstructorValuesFromBinaryMembers(targetType, invoke.Method.Arguments, binaryMembers));
			return true;
		}
		var values = new ValueInstance[members.Count];
		for (var parameterIndex = 0; parameterIndex < invoke.Method.Method.Parameters.Count; parameterIndex++)
		{
			//wtf, another 1.2m calls?
			var parameter = invoke.Method.Method.Parameters[parameterIndex];
			var memberIndex = FindMemberIndex(members, parameter.Name);
			if (memberIndex == -1)
				continue;
			var memberInitialValue = members[memberIndex].InitialValue;
			values[memberIndex] = parameterIndex < invoke.Method.Arguments.Count
				? EvaluateExpression(invoke.Method.Arguments[parameterIndex])
				: parameter.DefaultValue != null
					? EvaluateExpression(parameter.DefaultValue)
					: memberInitialValue != null
						? EvaluateExpression(memberInitialValue)
						: hasBinaryMembers && TryGetBinaryMemberInitialValue(binaryMembers, memberIndex,
							out var initialValue)
							? initialValue
							: CreateDefaultValue(members[memberIndex].Type);
		}
		for (var memberIndex = 0; memberIndex < members.Count; memberIndex++)
			if (!values[memberIndex].HasValue)
			{
				var memberInitialValue = members[memberIndex].InitialValue;
				values[memberIndex] = members[memberIndex].Type.IsTrait
					? CreateTraitInstance(members[memberIndex].Type)
					: memberInitialValue != null
						? EvaluateExpression(memberInitialValue)
						: hasBinaryMembers && TryGetBinaryMemberInitialValue(binaryMembers, memberIndex,
							out var initialValue)
							? initialValue
							: CreateDefaultValue(members[memberIndex].Type);
			}
		TryPreFillConstrainedListMembers(targetType, values);
		Memory.Registers[invoke.Register] = new ValueInstance(targetType, values);
		return true;
	}

	private static bool TryGetBinaryMemberInitialValue(IReadOnlyList<BinaryMember> binaryMembers,
		int memberIndex, out ValueInstance value)
	{
		if (memberIndex < binaryMembers.Count &&
			binaryMembers[memberIndex].InitialValueExpression is SetInstruction setInstruction)
		{
			value = setInstruction.ValueInstance;
			return true;
		}
		value = default;
		return false;
	}

	private static int FindMemberIndex(IReadOnlyList<Member> members, string name)
	{
		for (var index = 0; index < members.Count; index++)
			if (members[index].Name.Equals(name, StringComparison.OrdinalIgnoreCase))
				return index;
		return -1;
	}

	private ValueInstance[] CreateConstructorValuesFromBinaryMembers(Type targetType,
		IReadOnlyList<Expression> arguments, IReadOnlyList<BinaryMember> binaryMembers)
	{
		var values = new ValueInstance[binaryMembers.Count];
		var argumentIndex = 0;
		for (var memberIndex = 0; memberIndex < binaryMembers.Count; memberIndex++)
		{
			var memberType = targetType.FindType(binaryMembers[memberIndex].FullTypeName) ??
				targetType.FindType(GetShortTypeName(binaryMembers[memberIndex].FullTypeName));
			if (memberType is { IsTrait: true })
				values[memberIndex] = CreateTraitInstance(memberType);
			else if (argumentIndex < arguments.Count)
				values[memberIndex] = EvaluateExpression(arguments[argumentIndex++]);
			else if (memberType != null)
				values[memberIndex] = CreateDefaultComplexValue(memberType);
			else
				values[memberIndex] = new ValueInstance(executable.numberType, 0);
		}
		return values;
	}

	private void TryPreFillConstrainedListMembers(Type targetType, ValueInstance[] values)
	{
		var members = targetType.Members;
		for (var memberIndex = 0; memberIndex < members.Count; memberIndex++)
		{
			if (!values[memberIndex].IsList || values[memberIndex].List.Count > 0 ||
				members[memberIndex].Constraints == null)
				continue;
			var length = TryGetConstrainedLength(targetType, values, members[memberIndex]);
			if (length is not > 0)
				continue;
			var elementType = members[memberIndex].Type is GenericTypeImplementation genericList
				? genericList.ImplementationTypes[0]
				: members[memberIndex].Type;
			if (elementType.Members.FirstOrDefault(member => member.Name == Type.ElementsLowercase)?.Type is
				GenericTypeImplementation nestedElementsList)
				elementType = nestedElementsList.ImplementationTypes[0];
			var defaultElement = CreateDefaultComplexValue(elementType);
			values[memberIndex] = new ValueInstance(members[memberIndex].Type, defaultElement,
				length.Value);
		}
	}

	private static ValueInstance CreateDefaultValue(Type memberType)
	{
		if ((memberType.IsMutable
			? memberType.GetFirstImplementation()
			: memberType).IsList)
			return new ValueInstance(memberType, Array.Empty<ValueInstance>());
		if ((memberType.IsMutable
			? memberType.GetFirstImplementation()
			: memberType).IsDictionary)
			return new ValueInstance(memberType, new Dictionary<ValueInstance, ValueInstance>());
		if (memberType.IsText)
			return new ValueInstance("");
		if (memberType.IsBoolean)
			return new ValueInstance(memberType, false);
		if (memberType.IsNone)
			return new ValueInstance(memberType);
		if (memberType.Members.Count > 0 && !memberType.IsMutable)
			return new ValueInstance(memberType);
		if (memberType.IsMutable)
			// ReSharper disable once TailRecursiveCall
			return CreateDefaultValue(memberType.GetFirstImplementation());
		return new ValueInstance(memberType, 0);
	}

	private static ValueInstance CreateDefaultComplexValue(Type type)
	{
		if (type.IsList || type.IsDictionary || type.IsText || type.IsBoolean || type.IsNumber ||
			type.IsNone)
			return CreateDefaultValue(type);
		var members = type.Members;
		if (members.Count == 0)
			return CreateDefaultValue(type);
		var values = new ValueInstance[members.Count];
		for (var memberIndex = 0; memberIndex < members.Count; memberIndex++)
			values[memberIndex] = members[memberIndex].Type.IsTrait
				? CreateTraitInstance(members[memberIndex].Type)
				: members[memberIndex].InitialValue is Value initialValue
					? initialValue.Data
					: CreateDefaultValue(members[memberIndex].Type);
		return new ValueInstance(type, values);
	}

	private int? TryGetConstrainedLength(Type targetType, ValueInstance[] values, Member member)
	{
		foreach (var constraint in member.Constraints!)
		{
			if (constraint is not Binary { Method.Name: BinaryOperator.Is } binary ||
				binary.Instance?.ToString() != "Length")
				continue;
			var rhs = binary.Arguments[0];
			if (rhs is Value numberValue)
				return (int)numberValue.Data.Number;
			return TryEvaluateLengthInMemberScope(targetType, values, rhs) ??
				TryResolveMemberMethodLength(targetType, values, rhs);
		}
		return null;
	}

	private int? TryResolveMemberMethodLength(Type targetType, ValueInstance[] values,
		Expression rhs)
	{
		var rhsText = rhs.ToString();
		var dotIndex = rhsText.IndexOf('.');
		if (dotIndex <= 0)
			return null;
		var memberName = rhsText[..dotIndex];
		var methodName = rhsText[(dotIndex + 1)..];
		for (var memberIndex = 0; memberIndex < targetType.Members.Count; memberIndex++)
		{
			if (!targetType.Members[memberIndex].Name.Equals(memberName,
					StringComparison.OrdinalIgnoreCase) || !values[memberIndex].HasValue)
				continue;
			var memberValue = values[memberIndex];
			var typeInstance = memberValue.TryGetValueTypeInstance();
			var method = typeInstance?.ReturnType.FindMethod(methodName, []);
			if (method == null)
				continue;
			var methodInstructions = GetPrecompiledMethodInstructions(method);
			if (methodInstructions != null)
			{
				var childScope = InitializeChildScope();
				Memory.Frame.Set(Type.ValueLowercase, memberValue, isMember: true);
				TrySetScopeMembersFromTypeMembers(typeInstance!);
				RunInstructions(methodInstructions);
				var result = Returns;
				CleanupChildScope(childScope);
				if (result.HasValue)
					return (int)result.Value.Number;
			}
			else
			{
				var bodyResult = TryEvaluateMethodBodyWithInstance(method, memberValue, typeInstance!);
				if (bodyResult != null)
					return bodyResult;
			}
		}
		return null;
	}

	private int? TryEvaluateMethodBodyWithInstance(Method method, ValueInstance memberValue,
		ValueTypeInstance typeInstance)
	{
		try
		{
			var body = method.GetBodyAndParseIfNeeded();
			if (body is not Body { Expressions.Count: > 0 } methodBody)
				return null;
			var savedFrame = Memory.Frame;
			Memory.Frame = new CallFrame();
			Memory.Frame.Set(Type.ValueLowercase, memberValue, isMember: true);
			TrySetScopeMembersFromTypeMembers(typeInstance);
			try
			{
				var lastExpression = methodBody.Expressions[^1];
				return (int)EvaluateExpression(lastExpression).Number;
			}
			finally
			{
				Memory.Frame = savedFrame;
			}
		}
		catch
		{
			return null;
		}
	}

	private int? TryEvaluateLengthInMemberScope(Type targetType, ValueInstance[] values,
		Expression lengthExpression)
	{
		var savedFrame = Memory.Frame;
		Memory.Frame = new CallFrame();
		try
		{
			for (var memberIndex = 0; memberIndex < targetType.Members.Count; memberIndex++)
				if (values[memberIndex].HasValue)
					Memory.Frame.Set(targetType.Members[memberIndex].Name, values[memberIndex],
						isMember: true);
			return (int)EvaluateExpression(lengthExpression).Number;
		}
		catch
		{
			return null;
		}
		finally
		{
			Memory.Frame = savedFrame;
		}
	}

	private static ValueInstance CreateTraitInstance(Type traitType)
	{
		var concreteType = traitType.FindType(traitType.Name is Type.TextWriter or Type.Logger
			? Type.System
			: traitType.Name);
		return concreteType != null
			? new ValueInstance(concreteType, Array.Empty<ValueInstance>())
			: new ValueInstance(traitType, 0);
	}
}
