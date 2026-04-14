using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;
using Strict.HighLevelRuntime;

namespace Strict;

public sealed partial class VirtualMachine
{
	private bool TryHandleFromConstructor(Invoke invoke, Type targetType)
	{
		if (targetType is GenericTypeImplementation)
			return false;
		var info = invoke.MethodInfo;
		var members = targetType.Members;
		var hasBinaryMembers = TryGetBinaryMembers(targetType, out var binaryMembers);
		if (members.Count == 0 && hasBinaryMembers)
		{
			Memory.Registers[invoke.Register] = new ValueInstance(targetType,
				CreateConstructorValuesFromBinaryMembers(targetType, info.ArgumentRegisters,
					binaryMembers));
			return true;
		}
		var values = new ValueInstance[members.Count];
		for (var parameterIndex = 0; parameterIndex < info.ParameterNames.Length; parameterIndex++)
		{
			var parameterName = info.ParameterNames[parameterIndex];
			var memberIndex = FindMemberIndex(members, parameterName);
			if (memberIndex == -1 && parameterIndex < members.Count)
				memberIndex = parameterIndex;
			if (memberIndex == -1 || values[memberIndex].HasValue)
				continue;
			values[memberIndex] = parameterIndex < info.ArgumentRegisters.Length
				? Memory.Registers[info.ArgumentRegisters[parameterIndex]]
				: GetMemberInitialOrDefaultValue(members[memberIndex], hasBinaryMembers, binaryMembers,
					memberIndex);
		}
		for (var memberIndex = 0; memberIndex < members.Count &&
			memberIndex < info.ArgumentRegisters.Length; memberIndex++)
			if (!values[memberIndex].HasValue)
				values[memberIndex] = Memory.Registers[info.ArgumentRegisters[memberIndex]];
		for (var memberIndex = 0; memberIndex < members.Count; memberIndex++)
			if (!values[memberIndex].HasValue)
				values[memberIndex] = members[memberIndex].Type.IsTrait
					? CreateTraitInstance(members[memberIndex].Type)
					: GetMemberInitialOrDefaultValue(members[memberIndex], hasBinaryMembers, binaryMembers,
						memberIndex);
		TryPreFillConstrainedListMembers(targetType, values);
		ConvertBytesToColorsIfNeeded(members, values);
		Memory.Registers[invoke.Register] = new ValueInstance(targetType, values);
		return true;
	}

	private void ConvertBytesToColorsIfNeeded(List<Member> members, ValueInstance[] values)
	{
		for (var memberIndex = 0; memberIndex < members.Count; memberIndex++)
		{
			if (!values[memberIndex].HasValue || !values[memberIndex].IsList)
				continue;
			var memberType = members[memberIndex].Type;
			if (memberType is not GenericTypeImplementation memberGeneric ||
				!memberGeneric.ImplementationTypes[0].Name.Equals("Color",
					StringComparison.OrdinalIgnoreCase))
				continue;
			var argValue = values[memberIndex];
			var argItems = argValue.List.Items;
			if (argItems.Count == 0 || argItems[0].TryGetValueTypeInstance() != null)
				continue;
			var colorType = memberGeneric.ImplementationTypes[0];
			var byteType = executable.basePackage.FindType("Byte") ??
				executable.basePackage.GetType(Language.Type.Number);
			values[memberIndex] = ConvertByteListToColorList(argItems, memberType, colorType,
				byteType);
		}
	}

	private static ValueInstance ConvertByteListToColorList(IReadOnlyList<ValueInstance> bytes,
		Type listType, Type colorType, Type byteType)
	{
		var colorCount = bytes.Count / 4;
		var colors = new ValueInstance[colorCount];
		for (var colorIndex = 0; colorIndex < colorCount; colorIndex++)
		{
			var red = (byte)Math.Clamp(bytes[colorIndex * 4].Number, 0, 255);
			var green = (byte)Math.Clamp(bytes[colorIndex * 4 + 1].Number, 0, 255);
			var blue = (byte)Math.Clamp(bytes[colorIndex * 4 + 2].Number, 0, 255);
			var alpha = colorIndex * 4 + 3 < bytes.Count
				? (byte)Math.Clamp(bytes[colorIndex * 4 + 3].Number, 0, 255)
				: (byte)255;
			colors[colorIndex] = new ValueInstance(colorType,
			[
				new ValueInstance(byteType, (double)red),
				new ValueInstance(byteType, (double)green),
				new ValueInstance(byteType, (double)blue),
				new ValueInstance(byteType, (double)alpha)
			]);
		}
		return new ValueInstance(listType, colors);
	}

	private static ValueInstance GetMemberInitialOrDefaultValue(Member member,
		bool hasBinaryMembers, List<BinaryMember> binaryMembers, int memberIndex) =>
		member.InitialValue is Value initialValue
			? initialValue.Data
			: hasBinaryMembers && TryGetBinaryMemberInitialValue(binaryMembers, memberIndex,
				out var binaryInitialValue)
				? binaryInitialValue
				: CreateDefaultValue(member.Type);

	private static bool TryGetBinaryMemberInitialValue(List<BinaryMember> binaryMembers,
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

	private static int FindMemberIndex(List<Member> members, string name)
	{
		for (var index = 0; index < members.Count; index++)
			if (members[index].Name.Equals(name, StringComparison.OrdinalIgnoreCase))
				return index;
		return -1;
	}

	private ValueInstance[] CreateConstructorValuesFromBinaryMembers(Type targetType,
		Register[] argumentRegisters, List<BinaryMember> binaryMembers)
	{
		var values = new ValueInstance[binaryMembers.Count];
		var argumentIndex = 0;
		for (var memberIndex = 0; memberIndex < binaryMembers.Count; memberIndex++)
		{
			var memberType = targetType.FindType(binaryMembers[memberIndex].FullTypeName) ??
				targetType.FindType(GetShortTypeName(binaryMembers[memberIndex].FullTypeName));
			if (memberType is { IsTrait: true })
				values[memberIndex] = CreateTraitInstance(memberType);
			else if (argumentIndex < argumentRegisters.Length)
				values[memberIndex] = Memory.Registers[argumentRegisters[argumentIndex++]];
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
			if (!values[memberIndex].IsList || values[memberIndex].List.Count > 0)
				continue;
			var length = members[memberIndex].Constraints != null
				? TryGetConstrainedLength(targetType, values, members[memberIndex])
				: null;
			length ??= TryGetLengthFromSizeLikeMember(targetType, values);
			if (length is not > 0)
				continue;
			var elementType = members[memberIndex].Type is GenericTypeImplementation genericList
				? genericList.ImplementationTypes[0]
				: members[memberIndex].Type;
			if (elementType.Members.FirstOrDefault(member => member.Name == Type.ElementsLowercase)?.
					Type is GenericTypeImplementation nestedElementsList)
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
		return memberType.IsMutable
			// ReSharper disable once TailRecursiveCall
			? CreateDefaultValue(memberType.GetFirstImplementation())
			: new ValueInstance(memberType, 0);
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
			return TryResolveMemberMethodLength(targetType, values, rhs);
		}
		return null;
	}

	private int? TryResolveMemberMethodLength(Type targetType, ValueInstance[] values,
		Expression rhs)
	{
		var rhsText = rhs.ToString();
		var dotIndex = rhsText.IndexOf('.');
		if (dotIndex <= 0)
			return TryResolveLengthFromFrameVariable(targetType, values, rhsText);
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
			{
				var fallbackLength = TryResolveLengthFromTypeMembers(typeInstance, methodName);
				if (fallbackLength.HasValue)
					return fallbackLength;
				continue;
			}
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
				continue;
			}
			var interpretedLength = TryResolveLengthByInterpreter(method, memberValue);
			if (interpretedLength.HasValue)
				return interpretedLength;
		}
		return null;
	}

	private static int? TryResolveLengthFromTypeMembers(ValueTypeInstance? typeInstance,
		string methodName)
	{
		if (typeInstance == null || !methodName.Equals("Length", StringComparison.OrdinalIgnoreCase))
			return null;
		return typeInstance.TryGetValue("Width", out var width) &&
			typeInstance.TryGetValue("Height", out var height)
			? (int)(width.Number * height.Number)
			: null;
	}

	private int? TryResolveLengthByInterpreter(Method method, ValueInstance memberValue)
	{
		constraintLengthInterpreter ??= new Interpreter(executable.basePackage, TestBehavior.Disabled);
		try
		{
			return (int)constraintLengthInterpreter.Execute(method, memberValue, []).Number;
		}
		catch
		{
			return null;
		}
	}

	private Interpreter? constraintLengthInterpreter;

	private static int? TryResolveLengthFromFrameVariable(Type targetType, ValueInstance[] values,
		string variableName)
	{
		for (var memberIndex = 0; memberIndex < targetType.Members.Count; memberIndex++)
			if (targetType.Members[memberIndex].Name.Equals(variableName,
					StringComparison.OrdinalIgnoreCase) && values[memberIndex].HasValue &&
				targetType.Members[memberIndex].Type.IsNumber)
				return (int)values[memberIndex].Number;
		return null;
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

	private static string GetShortTypeName(string fullTypeName)
	{
		var index = fullTypeName.LastIndexOf(Context.ParentSeparator);
		return index >= 0
			? fullTypeName[(index + 1)..]
			: fullTypeName;
	}

	private static int? TryGetLengthFromSizeLikeMember(Type targetType, ValueInstance[] values)
	{
		for (var memberIndex = 0; memberIndex < targetType.Members.Count; memberIndex++)
		{
			if (!values[memberIndex].HasValue)
				continue;
			var typeInstance = values[memberIndex].TryGetValueTypeInstance();
			if (typeInstance == null)
				continue;
			if (!typeInstance.TryGetValue("Width", out var width) ||
				!typeInstance.TryGetValue("Height", out var height))
				continue;
			return (int)(width.Number * height.Number);
		}
		return null;
	}
}