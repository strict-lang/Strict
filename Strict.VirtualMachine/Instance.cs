using Type = Strict.Language.Type;

namespace Strict.VirtualMachine;

/// <summary>
/// The only place where we can have a "static" method call to one of the from methods of a type
/// before we have a type instance yet, it is the only way to create instances.
/// </summary>
public sealed class Instance
{
	public Instance(Type type, object value)
	{
		ReturnType = type;
		Value = value;
	}

	public Type ReturnType { get; }
	public object Value { get; }
}