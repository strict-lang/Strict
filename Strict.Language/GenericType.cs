using System.Linq;

namespace Strict.Language;

public sealed class GenericType : Type
{
	public GenericType(Type generic, Type implementation) : base(generic.Package,
		new TypeLines(generic.Name + "(" + implementation.Name + ")", Implement + generic.Name))
	{
		Generic = generic;
		Implementation = implementation;
		foreach (var methodsByNames in Generic.AvailableMethods)
		foreach (var method in methodsByNames.Value)
			if (method.ReturnType.IsGeneric || method.Parameters.Any(p => p.Type.IsGeneric))
				methods.Add(method.CloneWithImplementation(this));
	}

	public Type Generic { get; }
	public Type Implementation { get; }
}