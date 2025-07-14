using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Expressions;

public class Instance(Type type) : Expression(type)
{
	public static Expression Parse(Body body, Method method)
	{
		var valueInstance = new Instance((Type)method.Parent);
		body.AddVariable(Base.Value, valueInstance, false);
		return valueInstance;
	}

	public override string ToString() => Base.Value;
}