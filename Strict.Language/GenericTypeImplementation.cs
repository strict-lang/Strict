using System;
using System.Collections.Generic;
using System.Linq;

namespace Strict.Language;

public sealed class GenericTypeImplementation : Type
{
	public GenericTypeImplementation(Type generic, IReadOnlyList<Type> implementationTypes) : base(generic.Package,
		new TypeLines(generic.GetImplementationName(implementationTypes), HasWithSpaceAtEnd + generic.Name))
	{
		CreatedBy = "Generic: " + generic + ", Implementations: " + implementationTypes.ToWordList() +
			", " + CreatedBy;
		Generic = generic;
		ImplementationTypes = implementationTypes;
		var implementationTypeIndex = 0;
		foreach (var member in Generic.Members)
			members.Add(member.Type.IsGeneric && member.Type.Name != Base.Iterator//TODO: remove all these Iterator and List hacks!
				? member.CloneWithImplementation(member.Type.Name == Base.List
					? this
					: implementationTypes[implementationTypeIndex++])
				: member);
		foreach (var methodsByNames in Generic.AvailableMethods)
		foreach (var method in methodsByNames.Value)
			if (method.IsPublic || method.Name.AsSpan().IsOperator())
			{
				var foundMethodAlready = false;
				foreach (var existingMethod in methods)
					if (existingMethod.IsSameMethodNameReturnTypeAndParameters(method))
						foundMethodAlready = true;
				if (!foundMethodAlready)
					methods.Add(method.CloneWithImplementation(this));
			}
	}

	public Type Generic { get; }
	public IReadOnlyList<Type> ImplementationTypes { get; }
}