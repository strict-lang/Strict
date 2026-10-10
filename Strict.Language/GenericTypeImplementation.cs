namespace Strict.Language;

public sealed class GenericTypeImplementation : Type
{
	public GenericTypeImplementation(Type generic, Type[] implementationTypes, string typeName) :
		base(GetPackage(generic, implementationTypes),
			new TypeLines(typeName, CreateHasLines(generic, implementationTypes)))
	{
		Generic = generic;
		ImplementationTypes = implementationTypes;
		if (Generic.IsError)
			typeKind = TypeKind.Error;
		if (Generic.IsList)
			typeKind = TypeKind.List;
		if (Generic.IsDictionary)
			typeKind = TypeKind.Dictionary;
		if (Generic.IsMutable)
			typeKind = ImplementationTypes[0].typeKind;
		ImplementMembers();
		ImplementMethods();
	}

	/// <summary>
	/// List(Color) lives next to Color (Strict/ImageProcessing/List(Color)) when Color is inside
	/// the generic's package, List(Number) stays Strict/List(Number), same named types never mix.
	/// ponytail: List.strict code resolves names there, a Range there would shadow Strict/Range.
	/// </summary>
	private static Package GetPackage(Type generic, IEnumerable<Type> implementationTypes)
	{
		foreach (var implementationType in implementationTypes)
			for (var parent = implementationType.Package.Parent; parent is Package package;
				parent = package.Parent)
				if (package == generic.Package)
					return implementationType.Package;
		return generic.Package;
	}

	private static string[] CreateHasLines(Type generic, Type[] implementationTypes) =>
		generic.IsMutable && implementationTypes[0].IsGeneric
			? [HasWithSpaceAtEnd + generic.Name, HasWithSpaceAtEnd + GenericUppercase]
			: [HasWithSpaceAtEnd + generic.Name];

	public Type Generic { get; }
	public IReadOnlyList<Type> ImplementationTypes { get; }

	private void ImplementMembers()
	{
		var implementationTypeIndex = 0;
		for (var index = 0; index < Generic.Members.Count; index++)
		{
			var member = Generic.Members[index];
			if ((member.Type.IsGeneric || member.Type is GenericType) && member.Type.Name != Iterator)
				member = member.CloneWithImplementation(GetImplementedMemberType(member.Type,
					ref implementationTypeIndex));
			members.Add(member);
		}
	}

	private Type GetImplementedMemberType(Type memberType, ref int implementationTypeIndex)
	{
		if (memberType is GenericType { Generic.Name: List, GenericImplementations.Count: > 1 } generic)
			return generic.Generic.GetGenericImplementation(
				generic.Generic.GetGenericImplementation(ImplementationTypes[0]));
		return memberType.IsList
			? this
			: ImplementationTypes[implementationTypeIndex++];
	}

	private void ImplementMethods()
	{
		foreach (var methodsByNames in Generic.AvailableMethods)
		foreach (var method in methodsByNames.Value)
			// Do not copy from constructor (might be different with generics now implemented)
			if (method.Name != Method.From && (method.IsPublic || method.Name.AsSpan().IsOperator()))
			{
				var foundMethodAlready = false;
				foreach (var existingMethod in methods)
					if (existingMethod.IsSameMethodNameReturnTypeAndParameters(method))
					{ //ncrunch: no coverage start, no usecase yet
						foundMethodAlready = true;
						break;
					} //ncrunch: no coverage end
				if (!foundMethodAlready)
					methods.Add(new Method(method, this));
			}
	}

	internal void ReimplementMethods()
	{
		methods.Clear();
		ImplementMethods();
	}

	internal void ReimplementMembers()
	{
		members.Clear();
		cachedIteratorState = 0;
		cachedEvaluatedMemberTypes?.Clear();
		ImplementMembers();
	}
}