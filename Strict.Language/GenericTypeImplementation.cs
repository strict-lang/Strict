using System;
using System.Collections.Generic;

namespace Strict.Language;

public sealed class GenericTypeImplementation : Type
{
	public GenericTypeImplementation(Type generic, IReadOnlyList<Type> implementationTypes) : base(generic.Package,
		new TypeLines(GetTypeName(generic, implementationTypes), HasWithSpaceAtEnd + generic.Name))
	{
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
				methods.Add(method.CloneWithImplementation(this));
	}

	//TODO: why is there 2 ways to generate the type name, why not internally only keep one true way: List(Number), we can probably also remove the strange special rule in Context.FindType:
		//if (name.StartsWith(Base.List + DoubleOpenBrackets, StringComparison.Ordinal))
	private static string GetTypeName(Type generic, IReadOnlyList<Type> implementationTypes) =>
		generic.Name == Base.List && !implementationTypes[0].Name.EndsWith(')')
			? implementationTypes[0].Name.Pluralize()
			: generic.Name + implementationTypes.ToBrackets();

	public Type Generic { get; }
	public IReadOnlyList<Type> ImplementationTypes { get; }
}