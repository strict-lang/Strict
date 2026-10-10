using System.Collections.Concurrent;

namespace Strict.Language;

public partial class Type
{
	/// <summary>
	/// Any non-public member is automatically iterable if it has Iterator, for example, Text.strict
	/// or Error.strict have public members you have to iterate over yourself. If there are more
	/// private iterators, pick the first member automatically. List and number are also iterable.
	/// </summary>
	public bool IsIterator =>
		typeKind == TypeKind.Iterator || Name.StartsWith(Iterator + "(", StringComparison.Ordinal) ||
		HasAnyIteratorMember();

	private bool HasAnyIteratorMember()
	{
		var cached = Volatile.Read(ref cachedIteratorState);
		if (cached != 0)
			return cached == IteratorTrue;
		var computed = ExecuteIsIteratorCheck();
		Volatile.Write(ref cachedIteratorState, computed
			? IteratorTrue
			: IteratorFalse);
		return computed;
	}

	private bool ExecuteIsIteratorCheck()
	{
		CheckIfParsed();
		var evaluatedMemberTypes = cachedEvaluatedMemberTypes ??= new();
		foreach (var member in members)
		{
			if (evaluatedMemberTypes.TryGetValue(member.Type.Name, out var result))
				return result; //ncrunch: no coverage
			var isIterator = member is { IsPublic: false, Type.IsIterator: true };
			evaluatedMemberTypes[member.Type.Name] = isIterator;
			if (isIterator)
				return true;
		}
		return false;
	}

	/// <summary>
	/// Can OUR type be converted to sameOrUsableType and be used as such? Be careful how this is
	/// called. A derived RedApple can be used as the base class Apple, but not the other way around.
	/// </summary>
	public bool IsSameOrCanBeUsedAs(Type sameOrUsableType, bool allowImplicitConversion = true,
		int maxDepth = 2)
	{
		if (this == sameOrUsableType || sameOrUsableType.IsAny || (typeKind < TypeKind.List &&
			typeKind == sameOrUsableType.typeKind))
			return true;
		if (IsGenericTypeCompatible(sameOrUsableType))
			return true;
		if (allowImplicitConversion && IsImplicitAnyToConversion(sameOrUsableType))
			return true;
		if (IsEnum && members[0].Type.IsSameOrCanBeUsedAs(sameOrUsableType))
			return true;
		if ((IsMutable && GetFirstImplementation().IsSameOrCanBeUsedAs(sameOrUsableType)) ||
			(sameOrUsableType.IsMutable &&
				IsSameOrCanBeUsedAs(sameOrUsableType.GetFirstImplementation())))
			return true;
		if (HasExactlyOneMemberOfType(sameOrUsableType))
			return true;
		if (IsCompatibleOneOfType(sameOrUsableType))
			return true;
		if (DeclaresForIterator(sameOrUsableType))
			return true;
		return maxDepth >= 0 &&
			HasExactlyOneUsableMember(sameOrUsableType, allowImplicitConversion, maxDepth);
	}

	/// <summary>
	/// Checks whether this type can be adapted to targetType via existing to/from conversions.
	/// </summary>
	public bool CanBeConvertedTo(Type targetType, bool allowImplicitConversion = false)
	{
		if (IsSameOrCanBeUsedAs(targetType, allowImplicitConversion))
			return true;
		if (CanConvertBetweenByteListAndCompositeByteList(targetType))
			return true;
		if (targetType.CanBeCreatedFromSingleMember(this, allowImplicitConversion))
			return true;
		if (IsBaseTypeExcludedFromImplicitListConversion() ||
			targetType.IsBaseTypeExcludedFromImplicitListConversion())
			return false;
		if (AvailableMethods.TryGetValue(BinaryOperator.To, out var toMethods) &&
			toMethods.Any(method => method.ReturnType == targetType ||
				method.ReturnType.IsSameOrCanBeUsedAs(targetType, allowImplicitConversion)))
			return true;
		return targetType.AvailableMethods.TryGetValue(Method.From, out var fromMethods) &&
			fromMethods.Any(method => method.Parameters.Count == 1 &&
				IsSameOrCanBeUsedAs(method.Parameters[0].Type, allowImplicitConversion));
	}

	private bool IsBaseTypeExcludedFromImplicitListConversion() =>
		typeKind < TypeKind.List || Name == "Byte";

	private bool CanConvertBetweenByteListAndCompositeByteList(Type targetType)
	{
		if (this is not GenericTypeImplementation { Generic.IsList: true } sourceList ||
			targetType is not GenericTypeImplementation { Generic.IsList: true } targetList)
			return false;
		var sourceElement = sourceList.ImplementationTypes[0];
		var targetElement = targetList.ImplementationTypes[0];
		return (IsTypeComposedOfBytesOnly(sourceElement) && targetElement.Name == "Byte") ||
			(IsTypeComposedOfBytesOnly(targetElement) && sourceElement.Name == "Byte");
	}

	private static bool IsTypeComposedOfBytesOnly(Type type) =>
		type.Members.Count > 0 && type.Members.All(member => member.Type.Name is "Byte" or "Number");

	/// <summary>
	/// Returns true when this type explicitly declares "for Iterator(T)" and the target is that
	/// same Iterator(T). This is used to recognize that Range IS an Iterator(Number) because Range
	/// declares "for Iterator(Number)" in its method list.
	/// </summary>
	private bool DeclaresForIterator(Type targetType) =>
		targetType is GenericTypeImplementation { Generic.typeKind: TypeKind.Iterator } &&
		methods.Any(m => m.Name == "for" && m.ReturnType == targetType);

	private bool IsGenericTypeCompatible(Type sameOrUsableType)
	{
		if (this is GenericTypeImplementation sourceImplementation)
			return IsSourceGenericImplementationCompatible(sourceImplementation, sameOrUsableType);
		if (sameOrUsableType is GenericTypeImplementation targetImplementation)
			return targetImplementation.Generic == this;
		return false;
	}

	private static bool IsSourceGenericImplementationCompatible(
		GenericTypeImplementation sourceImplementation, Type sameOrUsableType)
	{
		if (sourceImplementation.Generic == sameOrUsableType)
			return true;
		if (sameOrUsableType is not GenericTypeImplementation targetImplementation ||
			sourceImplementation.Generic != targetImplementation.Generic ||
			sourceImplementation.ImplementationTypes.Count !=
			targetImplementation.ImplementationTypes.Count)
			return false;
		for (var implementationIndex = 0;
			implementationIndex < sourceImplementation.ImplementationTypes.Count; implementationIndex++)
			if (!sourceImplementation.ImplementationTypes[implementationIndex].CanBeConvertedTo(
				targetImplementation.ImplementationTypes[implementationIndex]))
				return false;
		return true;
	}

	private bool HasExactlyOneMemberOfType(Type targetType)
	{
		// Basically members.Count(m => m.Type == targetType) == 1, but more performant
		var found = false;
		foreach (var m in members)
			if (m.Type == targetType)
			{
				if (found)
					return false;
				found = true;
			}
		return found;
	}

	internal bool CanUseInheritedSingleMemberReturn(Type methodType, Type returnType) =>
		TryGetSingleValueMemberType(out var memberType) && methodType == memberType &&
		returnType.IsSameOrCanBeUsedAs(memberType, false, 1);

	private bool TryGetSingleValueMemberType(out Type memberType)
	{
		memberType = null!;
		for (var memberIndex = 0; memberIndex < members.Count; memberIndex++)
			if (!members[memberIndex].IsConstant)
			{
				// ReSharper disable once ConditionIsAlwaysTrueOrFalseAccordingToNullableAPIContract
				if (memberType != null)
					return false;
				memberType = members[memberIndex].Type;
			}
		// ReSharper disable once ConditionIsAlwaysTrueOrFalseAccordingToNullableAPIContract
		return memberType != null;
	}

	private bool HasExactlyOneUsableMember(Type targetType, bool allowImplicitConversion,
		int maxDepth)
	{
		var key = (targetType, allowImplicitConversion, maxDepth);
		var usableMemberCache = this.usableMemberCache ??= new();
		if (usableMemberCache.TryGetValue(key, out var cached))
			return cached;
		var found = false;
		for (var memberIndex = 0; memberIndex < members.Count; memberIndex++)
			if (!members[memberIndex].IsConstant && members[memberIndex].Type.
				IsSameOrCanBeUsedAs(targetType, allowImplicitConversion, maxDepth - 1))
			{
				if (found)
					return usableMemberCache[key] = false;
				found = true;
			}
		return usableMemberCache[key] = found;
	}

	private ConcurrentDictionary<(Type, bool, int), bool>? usableMemberCache;

	/// <summary>
	/// Only allow implicit conversions as defined in Any.strict (to Text, to Type, to HashCode)
	/// </summary>
	private static bool IsImplicitAnyToConversion(Context targetType) =>
		targetType.Name is Text or nameof(Type) or HashCode;

	private bool IsCompatibleOneOfType(Type sameOrBaseType)
	{
		if (sameOrBaseType is OneOfType oneOfType)
			for (var index = 0; index < oneOfType.Types.Length; index++)
				if (IsSameOrCanBeUsedAs(oneOfType.Types[index]))
					return true;
		return false;
	}

	/// <summary>
	/// When two types are using in a conditional expression, i.e., then and else return types and
	/// both are not based on each other, find the common base type that works for both.
	/// </summary>
	public Type? FindFirstUnionType(Type elseType)
	{
		if (elseType.IsError)
			return this;
		if (IsError)
			return elseType;
		// Allow number and iterators for return types
		if (Name == Number && elseType.IsIterator)
			return elseType;
		if (elseType.IsNumber && IsIterator)
			return this;
		foreach (var member in members)
			if (elseType.members.Any(otherMember => otherMember.Type == member.Type))
				return member.Type;
		foreach (var member in members)
		{
			if (member.Type == this)
				continue;
			var subUnionType = member.Type.FindFirstUnionType(elseType);
			if (subUnionType != null)
				return subUnionType;
		}
		foreach (var otherMember in elseType.members)
		{
			var otherSubUnionType = otherMember.Type.FindFirstUnionType(this);
			if (otherSubUnionType != null)
				return otherSubUnionType;
		}
		return null;
	}

	public bool IsUpcastable(Type otherType) =>
		IsEnum && otherType.IsEnum && otherType.Members.Any(member =>
			member.Name.Equals(Name, StringComparison.OrdinalIgnoreCase));
}
