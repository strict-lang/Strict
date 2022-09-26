namespace Strict.Language;

// ReSharper disable once HollowTypeName
public sealed class GenericType : Type
{
	public GenericType(Type generic, Type implementation) : base(generic.Package,
		new TypeLines(generic.Name + implementation.Name))
	{
		Generic = generic;
		Implementation = implementation;
	}

	public Type Generic { get; }
	public Type Implementation { get; }
}