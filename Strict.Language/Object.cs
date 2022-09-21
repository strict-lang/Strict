/*https://deltaengine.fogbugz.com/f/cases/25970
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

	public override string ToString() =>
		Type.Name + (GenericImplementation != null
			? "(" + GenericImplementation.Name + ")"
			: "");
}
*/