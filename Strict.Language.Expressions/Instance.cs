namespace Strict.Language.Expressions;

public class Instance : Expression
{
	public Instance(Type type) : base(type) { }

	public static Expression Parse(Body body, Method method)
	{
		var valueInstance = new Instance((Type)method.Parent);
		body.AddVariable(Base.Value, valueInstance);
		return valueInstance;
	}

	public override string ToString() => Base.Value;
}