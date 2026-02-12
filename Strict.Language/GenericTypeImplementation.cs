namespace Strict.Language;

public sealed class GenericTypeImplementation : Type
{
	public GenericTypeImplementation(Type generic, IReadOnlyList<Type> implementationTypes) : base(
		generic.Package, new TypeLines(generic.GetImplementationName(implementationTypes),
			CreateHasLines(generic, implementationTypes)))
	{
		CreatedBy = "Generic: " + generic + ", Implementations: " + implementationTypes.ToWordList() +
			", " + CreatedBy;
		Generic = generic;
		ImplementationTypes = implementationTypes;
		ImplementMembers();
		ImplementMethods();
	}

	private static string[] CreateHasLines(Type generic, IReadOnlyList<Type> implementationTypes) =>
		generic.IsMutable && implementationTypes[0].IsGeneric
			? [HasWithSpaceAtEnd + generic.Name, HasWithSpaceAtEnd + Base.Generic]
			: [HasWithSpaceAtEnd + generic.Name];

	public Type Generic { get; }
	public IReadOnlyList<Type> ImplementationTypes { get; }

	private void ImplementMembers()
	{
		var implementationTypeIndex = 0;
		foreach (var member in Generic.Members)
			members.Add(
				member.Type.IsGeneric &&
				member.Type.Name != Base.Iterator
					? member.CloneWithImplementation(member.Type.Name == Base.List
						? this
						: ImplementationTypes[implementationTypeIndex++])
					: member);
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
}