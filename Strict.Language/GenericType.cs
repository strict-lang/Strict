using System.Collections.Generic;
using System.Linq;

namespace Strict.Language;

public sealed class GenericType : Type
{
	public GenericType(Type generic, IReadOnlyList<Type> implementationTypes) : base(generic.Package,
		new TypeLines(GetTypeName(generic, implementationTypes), Implement + generic.Name))
	{
		Generic = generic;
		ImplementationTypes = implementationTypes;
		foreach (var methodsByNames in Generic.AvailableMethods)
		foreach (var method in methodsByNames.Value)
			methods.Add(method.CloneWithImplementation(this));
	}

	private static string GetTypeName(Type generic, IReadOnlyList<Type> implementationTypes) =>
		generic.IsList
			? implementationTypes[0].Name.MakeItPlural()
			: generic.Name + implementationTypes.ToBrackets();

	public Type Generic { get; }
	public IReadOnlyList<Type> ImplementationTypes { get; }
}