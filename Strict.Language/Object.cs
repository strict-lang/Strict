/*TODO: maybe later? left over from GenericType refactoring
namespace Strict.Language;

public class Object
{
	public Object(Type type, Type? genericImplementation)
	{
		Type = type;
		GenericImplementation = genericImplementation;
	}

	public Type Type { get; }
	public Type? GenericImplementation { get; }
	//TODO: we can also store value this currently has, lines it was defined and used, etc.
	//TODO: is mutable, can be reduced/constant from now on, etc.

	public override string ToString() =>
		Type.Name + (GenericImplementation != null
			? "(" + GenericImplementation.Name + ")"
			: "");
}
*/